"""Bizz01 Aegis: fast, deterministic, placement-oriented Mech Royale agent.

No third-party dependencies; uses the provided mechfight/1 wrapper.
The bot never reads hidden match information. It uses only hello.map and observations.
"""
import array
import heapq
import json
import math
import os
import random

import mechfight as mf

PI = math.pi
TAU = PI * 2


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm(x, y):
    d = math.hypot(x, y)
    return (x / d, y / d) if d > 1e-7 else (0.0, 0.0)


def angle_delta(a, b):
    return (a - b + PI) % TAU - PI


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segdist2(x, y, a, b, c, d):
    dx, dy = c - a, d - b
    length = dx * dx + dy * dy
    t = clamp(((x - a) * dx + (y - b) * dy) / length, 0.0, 1.0) if length > 1e-8 else 0.0
    return (x - (a + t * dx)) ** 2 + (y - (b + t * dy)) ** 2


def poly_near(x, y, ring, padding):
    inside = False
    mind2 = padding * padding
    a, b = ring[-2], ring[-1]
    for i in range(0, len(ring), 2):
        c, d = ring[i], ring[i + 1]
        if ((b > y) != (d > y)) and x < (c - a) * (y - b) / (d - b) + a:
            inside = not inside
        if segdist2(x, y, a, b, c, d) <= mind2:
            return True
        a, b = c, d
    return inside


class NavGrid:
    """Conservative 2-D footprint navigation. Precomputed once in hello budget."""
    STEP = 7.5

    def __init__(self, directory):
        self.ok = False
        self.grid = bytearray()
        try:
            with open(os.path.join(directory, 'v2', 'world.json'), encoding='utf-8') as f:
                world = json.load(f)
            terrain = world['terrain']
            self.tx, self.ty = terrain['minX'], terrain['minY']
            self.tnx, self.tny = terrain['nx'], terrain['ny']
            self.ts = terrain['step']
            with open(os.path.join(directory, 'v2', 'terrain.i16'), 'rb') as f:
                self.heights = array.array('h')
                self.heights.frombytes(f.read())
            if __import__('sys').byteorder != 'little':
                self.heights.byteswap()
            bounds = world['bounds']
            self.x0, self.y0 = bounds['minX'], bounds['minY']
            self.nx = int((bounds['maxX'] - self.x0) / self.STEP) + 1
            self.ny = int((bounds['maxY'] - self.y0) / self.STEP) + 1
            self.grid = bytearray(self.nx * self.ny)
            # Keep mech collision cylinder away from map bounds.
            for j in range(self.ny):
                for i in range(self.nx):
                    x, y = self.xy(i, j)
                    if (x < bounds['minX'] + 7 or x > bounds['maxX'] - 7
                            or y < bounds['minY'] + 7 or y > bounds['maxY'] - 7):
                        self.grid[j * self.nx + i] = 1
            for prism in world['prisms']:
                ring = prism['ring']
                if len(ring) < 6:
                    continue
                xs, ys = ring[::2], ring[1::2]
                margin = 5.0
                ix0 = max(0, int((min(xs) - margin - self.x0) / self.STEP))
                ix1 = min(self.nx - 1, int((max(xs) + margin - self.x0) / self.STEP) + 1)
                iy0 = max(0, int((min(ys) - margin - self.y0) / self.STEP))
                iy1 = min(self.ny - 1, int((max(ys) + margin - self.y0) / self.STEP) + 1)
                for j in range(iy0, iy1 + 1):
                    for i in range(ix0, ix1 + 1):
                        idx = j * self.nx + i
                        if self.grid[idx]:
                            continue
                        x, y = self.xy(i, j)
                        ground = self.ground(x, y)
                        # Pass under high bridges, or walk over short steps.
                        if prism['base'] > ground + 11 or prism['top'] < ground + 1.0:
                            continue
                        if poly_near(x, y, ring, margin):
                            # Courtyards/holes can be traversed if sufficiently clear.
                            if any(poly_near(x, y, h, -1.0) for h in prism.get('holes', [])):
                                continue
                            self.grid[idx] = 1
            self.ok = True
        except (OSError, KeyError, ValueError, TypeError, OverflowError):
            self.ok = False

    def ground(self, x, y):
        i = int(clamp((x - self.tx) / self.ts, 0, self.tnx - 1))
        j = int(clamp((y - self.ty) / self.ts, 0, self.tny - 1))
        return self.heights[j * self.tnx + i] * 0.01

    def xy(self, i, j):
        return self.x0 + i * self.STEP, self.y0 + j * self.STEP

    def ij(self, x, y):
        return (int(clamp(round((x - self.x0) / self.STEP), 0, self.nx - 1)),
                int(clamp(round((y - self.y0) / self.STEP), 0, self.ny - 1)))

    def clear(self, a, b):
        if not self.ok:
            return True
        length = d2(a, b)
        if length < 1e-4:
            return True
        steps = max(1, int(length / (self.STEP * 0.55)))
        for k in range(1, steps + 1):
            u = k / steps
            i, j = self.ij(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
            if self.grid[j * self.nx + i]:
                return False
        return True

    def free_near(self, x, y):
        i, j = self.ij(x, y)
        if not self.grid[j * self.nx + i]:
            return i, j
        for radius in range(1, 9):
            options = []
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    a, b = i + di, j + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and not self.grid[b * self.nx + a]:
                        options.append((di * di + dj * dj, a, b))
            if options:
                _, a, b = min(options)
                return a, b
        return i, j

    def route(self, start, goal, enemies):
        if not self.ok or self.clear(start, goal):
            return [goal]
        si, sj = self.free_near(start[0], start[1])
        ti, tj = self.free_near(goal[0], goal[1])
        begin, end = sj * self.nx + si, tj * self.nx + ti
        openq = [(0.0, begin)]
        best = {begin: 0.0}
        parents = {}
        reached = None
        best_last, best_h = begin, math.hypot(ti - si, tj - sj)
        # Hard cap makes pathfinding comfortably sub-50ms, even on the biggest detour.
        for iteration in range(3300):
            if not openq:
                break
            f, cur = heapq.heappop(openq)
            ci, cj = cur % self.nx, cur // self.nx
            h = math.hypot(ti - ci, tj - cj)
            if h < best_h:
                best_last, best_h = cur, h
            if cur == end or h <= 1.1:
                reached = cur
                break
            cur_g = best[cur]
            for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                nxt = nj * self.nx + ni
                if self.grid[nxt]:
                    continue
                if di and dj and (self.grid[cj * self.nx + ni] or self.grid[nj * self.nx + ci]):
                    continue
                # Near visible mechs, take a safer detour where possible.
                penalty = 0.0
                if enemies:
                    wx, wy = self.xy(ni, nj)
                    for ex, ey in enemies[:5]:
                        dd = math.hypot(wx - ex, wy - ey)
                        if dd < 52:
                            penalty += (52 - dd) / 85
                cost = cur_g + (1.4142 if di and dj else 1.0) + penalty
                if cost >= best.get(nxt, 1e100):
                    continue
                best[nxt] = cost
                parents[nxt] = cur
                heapq.heappush(openq, (cost + math.hypot(ti - ni, tj - nj) * 1.08, nxt))
        tail = reached if reached is not None else best_last
        path = []
        while tail != begin and tail in parents and len(path) < 1000:
            path.append(self.xy(tail % self.nx, tail // self.nx))
            tail = parents[tail]
        path.reverse()
        if reached is not None and self.clear(path[-1] if path else start, goal):
            path.append(goal)
        return path or [goal]


class Bot:
    name = 'Bizz01-Aegis-v2'

    def __init__(self, hello: mf.Hello):
        self.config = hello['config']
        self.nav = NavGrid(hello.get('map', {}).get('dir', '/arena'))
        self.rng = random.Random(hello.get('seed', 0))
        self.route_path = []
        self.route_goal = None
        self.route_at = -100
        self.goal_type = None
        self.last_pos = None
        self.stuck_since = None
        self.jet_until = 0.0
        self.orbit_dir = -1 if hello.get('slot', 0) % 2 else 1
        self.last_enemy = None
        self.scan_phase = 0

    def choose_upgrade(self, choices, hp_ratio, weapons):
        base = {
            'nanites': 110 if hp_ratio < .96 else 90, 'armor': 105,
            'plating': 101 if hp_ratio < .72 else 82,
            'servos': 99, 'jets': 91, 'reactor': 77,
            'capacitor': 72, 'laser_damage': 91, 'laser_charge': 95,
            'gun_damage': 76, 'gun_rate': 74,
            'lightning_damage': 85, 'lightning_range': 72,
            'mortar_reload': 65, 'mortar_splash': 70,
            'gyros': 65, 'sensors': 30,
        }
        if not weapons[1]:
            base['gun_damage'] = 85
        return max(range(len(choices)), key=lambda i: base.get(choices[i].get('id'), 40))

    def choose_goal(self, obs, enemies, me):
        x, y, z = me['pos']
        here = (x, y)
        t = obs['t']
        zone = obs['zone']
        nextc = zone['next_center']
        nr = zone['next_radius']
        distn = d2(here, nextc)
        speed = me.get('stats', {}).get('walk_speed', 11.7)
        time_to_shrink = max(0, zone['shrink_end'] - t)
        # A future safe zone is more important than chasing a tempting drop.
        safe_margin = max(15.0, min(45.0, nr * .12))
        outside_future = distn > nr - safe_margin
        zone_urgent = (not me['in_zone'] or
                       (outside_future and (time_to_shrink < (distn - nr + safe_margin) / speed + 22
                                            or nr < 130)))
        near_enemy = min((d2(here, m['pos']) for m in enemies), default=1e6)
        high_threat = near_enemy < (140 if me['hp'] < .60 * me['max_hp'] else 105) or sum(d2(here, e['pos']) < 150 for e in enemies) >= 2

        # Opposite the hostile pack is safer than a predictable center-camping point.
        ax, ay, total = 0.0, 0.0, 0.0
        for enemy in enemies:
            dst = d2(here, enemy['pos'])
            if dst > 230:
                continue
            w = 1.0 / max(30, dst) ** 1.4
            ax += (nextc[0] - enemy['pos'][0]) * w
            ay += (nextc[1] - enemy['pos'][1]) * w
            total += w
        if total:
            ax, ay = norm(ax, ay)
        else:
            phase = (t / 45 + self.orbit_dir * 0.7)
            ax, ay = math.cos(phase), math.sin(phase)
        ring = min(max(0, nr - safe_margin - 12), 105, nr * .50)
        anchor = (nextc[0] + ring * ax, nextc[1] + ring * ay)
        # During earlier phases favor arriving to next zone, not the closing edge.
        if outside_future:
            return (anchor if nr < 220 else tuple(nextc)), 'zone', zone_urgent, high_threat

        # A well-timed supply drop provides a rare full heal; avoid contested rings.
        if not high_threat and not zone_urgent:
            for drop in obs.get('drops', []):
                dp = drop['pos']
                dd = d2(here, dp)
                nearby = sum(d2(dp, m['pos']) < 72 for m in enemies)
                if (dd < 105 and nearby == 0 and drop['lands_in'] < dd / max(speed, 1) + 6
                        and (me['hp'] < me['max_hp'] * .87 or me['cubes'] < 5)
                        and d2(dp, nextc) <= nr - 10):
                    return tuple(dp[:2]), 'drop', False, False

        # Loot only when it is not bait; pick up all convenient cubes and second weapons.
        loot = []
        if not high_threat and not zone_urgent:
            for cube in obs['cubes']:
                dd = d2(here, cube['pos'])
                if dd < 88 and d2(cube['pos'], nextc) < nr - 5:
                    hostile = min((d2(cube['pos'], m['pos']) for m in enemies), default=1e6)
                    if hostile >= 43:
                        loot.append((dd - 43, tuple(cube['pos'][:2]), 'cube'))
            existing = {w['kind'] for w in me['weapons'] if w}
            for pickup in obs['pickups']:
                kind = pickup['kind']
                if kind in existing or kind == 'gun':
                    continue
                dd = d2(here, pickup['pos'])
                if dd < 135 and d2(pickup['pos'], nextc) < nr - 4:
                    hostile = min((d2(pickup['pos'], m['pos']) for m in enemies), default=1e6)
                    if hostile >= 50:
                        priority = {'laser': 68, 'mortar': 57, 'lightning': 43}.get(kind, 0)
                        if me['weapons'][1] is None:
                            priority += 35
                        loot.append((dd - priority, tuple(pickup['pos'][:2]), 'weapon'))
            if loot:
                best = min(loot, key=lambda item: item[0])
                if best[0] < 95:
                    return best[1], best[2], False, False

        if high_threat:
            # Use the full safe circle to disengage, but don't flee outside it.
            sx, sy = 0.0, 0.0
            for enemy in enemies:
                dx, dy = x - enemy['pos'][0], y - enemy['pos'][1]
                dist = max(8.0, math.hypot(dx, dy))
                if dist < 140:
                    w = (140 - dist) / dist
                    sx += w * dx / dist
                    sy += w * dy / dist
            dx, dy = norm(sx, sy)
            if dx or dy:
                flee = (x + dx * (75 if near_enemy < 55 else 48),
                        y + dy * (75 if near_enemy < 55 else 48))
                maxrad = max(6.0, nr - max(15, safe_margin))
                fd = d2(flee, nextc)
                if fd > maxrad:
                    # Don't run into the death-zone while evading.
                    flee = (nextc[0] + (flee[0] - nextc[0]) * maxrad / fd,
                            nextc[1] + (flee[1] - nextc[1]) * maxrad / fd)
                if d2(flee, here) > 6:
                    return flee, 'evade', False, True
        if zone_urgent or d2(here, anchor) > 55:
            return anchor, 'zone', zone_urgent, high_threat
        # Don't stop entirely: orbit at low speed so distant bots miss lead shots.
        phase = t * .17 * self.orbit_dir
        return (x + 30 * math.cos(phase), y + 30 * math.sin(phase)), 'patrol', False, high_threat

    def aim_at(self, me, target, kind, dist):
        pos, vel = target['pos'], target.get('vel', [0, 0, 0])
        origin = me['pos']
        muzzle = me.get('stats', {}).get('muzzle_height', 7.0)
        zx = origin[2] + muzzle
        speed = me['weapons'][0]['stats'].get('speed', 252) if me['weapons'][0] else 252
        if kind == 'mortar':
            speed = 72
        if kind == 'gun':
            travel = min(.95, dist / max(30, speed))
        elif kind == 'mortar':
            travel = min(4.1, dist / 55.0)
        else:
            travel = .08
        px = pos[0] + vel[0] * travel
        py = pos[1] + vel[1] * travel
        pz = pos[2] + (1.7 if target['kind'] in ('npc', 'crate') else 4.5) + vel[2] * travel
        dx, dy = px - origin[0], py - origin[1]
        horizontal = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        dz = pz - zx
        if kind == 'mortar':
            wm = next((w for w in me['weapons'] if w and w['kind'] == 'mortar'), None)
            if wm:
                speed = wm['stats'].get('speed', speed)
            gravity = self.config['weapons']['mortar'].get('gravity', 22)
            discriminant = speed**4 - gravity * (gravity * horizontal**2 + 2 * dz * speed**2)
            if discriminant < 0 or horizontal < 20:
                return None
            pitch = math.atan((speed**2 - math.sqrt(discriminant)) / max(1, gravity * horizontal))
        else:
            pitch = math.atan2(dz, max(1, horizontal))
        return yaw, clamp(pitch, -0.68, 1.25)

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs['self']
        t = obs['t']
        x, y, z = me['pos']
        here = (x, y)
        hp_ratio = me['hp'] / max(1, me['max_hp'])
        enemies = [m for m in obs['mechs'] if m['kind'] == 'mech']
        enemies.sort(key=lambda e: d2(here, e['pos']))
        act = {'tick': obs['tick'], 'fire': [False, False]}

        if obs.get('offer'):
            act['pick'] = self.choose_upgrade(obs['offer']['choices'], hp_ratio, me['weapons'])

        # Replacing an inferior second weapon is possible but never drop the starter gun.
        owned = {w['kind'] for w in me['weapons'] if w}
        pickup_radius = self.config['loot']['pickup_radius']
        if me['weapons'][1] is None:
            candidates = [p for p in obs['pickups'] if p['kind'] != 'gun' and d2(here, p['pos']) <= pickup_radius - .3
                          and abs(p['pos'][2] - z) <= 3.5]
            if candidates:
                candidates.sort(key=lambda p: ({'laser': 0, 'mortar': 1, 'lightning': 2}.get(p['kind'], 9),
                                               d2(here, p['pos'])))
                act['take'] = candidates[0]['id']
        elif 'laser' not in owned:
            replacements = [p for p in obs['pickups'] if p['kind'] == 'laser'
                            and d2(here, p['pos']) < pickup_radius - .4
                            and abs(p['pos'][2] - z) <= 3.5]
            if replacements and not enemies:
                act['take'], act['drop'] = replacements[0]['id'], 1

        goal, goal_type, zone_urgent, threat = self.choose_goal(obs, enemies, me)
        # Reuse path only while roughly targeting the same place, to keep CPU headroom.
        target_moved = self.route_goal is None or d2(goal, self.route_goal) > 16
        recompute = (target_moved or t - self.route_at > 2.5 or not self.route_path)
        if recompute:
            hostile_coords = [tuple(e['pos'][:2]) for e in enemies]
            self.route_path = self.nav.route(here, goal, hostile_coords)
            self.route_goal = goal
            self.route_at = t
        # Skip waypoints already reached or visible from current location.
        while len(self.route_path) > 1 and d2(here, self.route_path[0]) < 10:
            self.route_path.pop(0)
        if self.nav.ok:
            for k in range(min(len(self.route_path) - 1, 8), 0, -1):
                if self.nav.clear(here, self.route_path[k]):
                    self.route_path = self.route_path[k:]
                    break
        waypoint = self.route_path[0] if self.route_path else goal
        mx, my = norm(waypoint[0] - x, waypoint[1] - y)
        # Don't march into visible enemies; sidestep into open space instead.
        if enemies:
            near_dist = d2(here, enemies[0]['pos'])
            if near_dist < 65:
                ex, ey = enemies[0]['pos'][:2]
                away_x, away_y = norm(x - ex, y - ey)
                dangerous = clamp((65 - near_dist) / 55, 0, .82)
                side = self.orbit_dir
                if len(enemies) > 1:
                    side = 1 if sum(e['pos'][0] - x for e in enemies[:3]) > 0 else -1
                mx, my = norm(mx * (1 - dangerous) + (away_x - side * away_y * .36) * dangerous,
                              my * (1 - dangerous) + (away_y + side * away_x * .36) * dangerous)
        # Dodge observed incoming projectiles by shifting laterally to their flight.
        # Use short-horizon closest approach, not simply proximity to a projectile.
        dodge_x, dodge_y = 0.0, 0.0
        for bullet in obs.get('projectiles', []):
            bp = bullet.get('pos', (0, 0, 0))
            bv = bullet.get('vel', (0, 0, 0))
            rx, ry = bp[0] - x, bp[1] - y
            vx, vy = bv[0] - me['vel'][0], bv[1] - me['vel'][1]
            v2 = vx * vx + vy * vy
            if v2 < 1:
                continue
            arrival = -(rx * vx + ry * vy) / v2
            if not 0.03 < arrival < .72:
                continue
            miss_x, miss_y = rx + vx * arrival, ry + vy * arrival
            miss = math.hypot(miss_x, miss_y)
            danger_radius = 12 if bullet.get('kind') == 'mortar' else 6.5
            if miss > danger_radius:
                continue
            nx, ny = norm(-vy, vx)
            sign = 1 if (miss_x * nx + miss_y * ny) < 0 else -1
            weight = (danger_radius - miss) / danger_radius * (1 - arrival * .55)
            dodge_x += nx * sign * weight
            dodge_y += ny * sign * weight
        if dodge_x or dodge_y:
            dx, dy = norm(dodge_x, dodge_y)
            mx, my = norm(mx * .45 + dx * .85, my * .45 + dy * .85)
        # Stuck detection (especially at roofs/steps) -> short jump-jet rescue.
        if self.last_pos is None:
            self.last_pos = (x, y, t)
        elif t - self.last_pos[2] >= 1.35:
            traveled = d2(here, self.last_pos)
            if traveled < 3.0 and d2(goal, here) > 17:
                self.jet_until = t + 2.7
                self.route_at = -100
            self.last_pos = (x, y, t)
        energy = me['energy']
        urgent_boost = zone_urgent or not me['in_zone'] or (threat and (hp_ratio < .92 or len(enemies) > 1)) or bool(dodge_x or dodge_y)
        boost = (energy > (11 if not me['in_zone'] else 25) and
                 (urgent_boost or (goal_type in ('weapon', 'cube', 'drop') and energy > 75 and d2(goal, here) > 40)))
        jet = t < self.jet_until and energy > 17
        act['move'] = [round(mx, 5), round(my, 5)]
        act['boost'] = bool(boost or (jet and not me['grounded']))
        if me['grounded'] and (jet or (threat and energy > 40 and int(t * 2.2) % 13 == 0)):
            act['jump'] = True

        # Placement-first combat policy. Shooting resets health regeneration; don't
        # trade shots at long range just because an enemy appears in the sensor list.
        # Only defend against immediate threats, punish very weak targets, or farm
        # crates/NPCs when safe. ALL weapon slots share one commanded torso aim.
        nearest = d2(here, enemies[0]['pos']) if enemies else 1e6
        hostile = None
        if enemies:
            for enemy in enemies:
                dist = d2(here, enemy['pos'])
                exposed = not self.nav.ok or self.nav.clear(here, enemy['pos'])
                weak = enemy.get('hp', 1400) < 240
                if exposed and (dist < (64 if hp_ratio < .48 else 92) or (weak and dist < 125)):
                    hostile = enemy
                    break
        can_farm = nearest > 145 and not zone_urgent and hp_ratio >= .80 and me['in_zone']
        target = hostile
        if target is None and can_farm:
            farm = []
            for npc in obs['mechs']:
                if npc['kind'] == 'npc':
                    distance = d2(here, npc['pos'])
                    if 18 < distance < 100 and (not self.nav.ok or self.nav.clear(here, npc['pos'])):
                        farm.append((distance - 12, npc))
            for crate in obs['crates']:
                distance = d2(here, crate['pos'])
                if distance < 90 and (not self.nav.ok or self.nav.clear(here, crate['pos'])):
                    c = {'kind': 'crate', 'pos': crate['pos'], 'vel': [0, 0, 0], 'hp': crate['hp'], 'id': crate['id']}
                    farm.append((distance, c))
            if farm:
                target = min(farm, key=lambda x: x[0])[1]

        w0, w1 = me['weapons']
        if target is not None:
            distance = d2(here, target['pos'])
            is_hostile = target['kind'] == 'mech'
            # Prefer precision hitscan at distance; mortar when in a safe firing
            # band; gun when second weapon can't be used. Laser and gun must not
            # both shoot at different pitch solutions in the same decision.
            secondary = None
            if w1:
                kind = w1['kind']
                if (kind == 'laser' and distance < w1['stats'].get('range', 280) - 5
                        and target['kind'] != 'crate' and (not threat or hp_ratio > .70)):
                    secondary = kind
                elif kind == 'lightning' and distance < min(42, w1['stats'].get('range', 45) - 2) and energy > 7:
                    secondary = kind
                elif kind == 'mortar' and 33 < distance < 185 and (is_hostile or target['kind'] == 'npc'):
                    secondary = kind
            active_kind = secondary or ('gun' if w0 and w0['kind'] == 'gun' else None)
            aim = self.aim_at(me, target, active_kind, distance) if active_kind else None
            if aim:
                act['aim'] = [round(aim[0], 5), round(aim[1], 5)]
                yaw_gap = abs(angle_delta(aim[0], me['aim'][0]))
                pitch_gap = abs(aim[1] - me['aim'][1])
                aligned = yaw_gap < .07 and pitch_gap < .08
                if secondary == 'laser':
                    charge = w1.get('charge', 0)
                    charge_goal = w1['stats'].get('charge', .75)
                    # Hold to charge, release only when aligned and substantially
                    # charged. Automatically release after hold cap.
                    act['fire'][1] = charge < charge_goal * .98 or not aligned
                    if charge >= charge_goal * 1.17:
                        act['fire'][1] = False
                elif secondary == 'lightning':
                    act['fire'][1] = yaw_gap < .17 and energy > 7
                elif secondary == 'mortar':
                    act['fire'][1] = aligned
                elif active_kind == 'gun':
                    act['fire'][0] = aligned and distance < w0['stats'].get('range', 230) - 5
        else:
            # Face a threatening mech to avoid taking rear-armor damage even when
            # retreating. Otherwise scan in broad arcs to discover safe loot.
            if enemies:
                e = enemies[0]
                yaw = math.atan2(e['pos'][1] - y, e['pos'][0] - x)
            else:
                yaw = math.atan2(my, mx) if mx or my else me['aim'][0]
                yaw += 0.55 * math.sin(t * .46)
            act['aim'] = [round(yaw, 5), 0.0]

        # Don't waste energy or risk slowing down for laser charging when fleeing.
        if w1 and w1['kind'] == 'laser' and w1.get('charge', 0) > 0:
            if target is None or (threat and hp_ratio < .65):
                act['fire'][1] = False
        if w0 and w0['kind'] == 'gun':
            ammo = w0.get('ammo', 12)
            if ammo <= 2 or (target is None and ammo < w0['stats'].get('mag', 12)):
                act['reload'] = True
        return act


mf.run(Bot)
