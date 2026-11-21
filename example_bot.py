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



class MapSight:
    """Exact-ish solid-prism LOS, unlike 2D navigation which cannot see over walls."""
    def __init__(self, directory):
        self.prisms = []
        try:
            with open(os.path.join(directory, 'v2', 'world.json'), encoding='utf8') as f:
                world = json.load(f)
            for p in world['prisms']:
                r = p['ring']
                if len(r) < 6:
                    continue
                xs, ys = r[::2], r[1::2]
                self.prisms.append((min(xs), max(xs), min(ys), max(ys), p['base'], p['top'], r, p.get('holes', [])))
        except (OSError, ValueError, KeyError):
            pass

    @staticmethod
    def inside(px, py, ring):
        inside = False
        ax, ay = ring[-2:]
        for i in range(0, len(ring), 2):
            bx, by = ring[i], ring[i + 1]
            if (ay > py) != (by > py) and px < (bx - ax) * (py - ay) / (by - ay) + ax:
                inside = not inside
            ax, ay = bx, by
        return inside

    def clear(self, a, b):
        if not self.prisms:
            return True
        ax, ay, az = a
        bx, by, bz = b
        dx, dy, dz = bx - ax, by - ay, bz - az
        left, right = min(ax, bx), max(ax, bx)
        bottom, top = min(ay, by), max(ay, by)
        minz, maxz = min(az, bz), max(az, bz)
        for xmin, xmax, ymin, ymax, low, high, ring, holes in self.prisms:
            if (xmax < left or xmin > right or ymax < bottom or ymin > top
                    or high < minz or low > maxz):
                continue
            ts = [0.0, 1.0]
            cx, cy = ring[-2:]
            for j in range(0, len(ring), 2):
                ex, ey = ring[j], ring[j + 1]
                vx, vy = ex - cx, ey - cy
                den = dx * vy - dy * vx
                if abs(den) > 1e-9:
                    wx, wy = cx - ax, cy - ay
                    t = (wx * vy - wy * vx) / den
                    u = (wx * dy - wy * dx) / den
                    if 0 < t < 1 and 0 <= u <= 1:
                        ts.append(t)
                cx, cy = ex, ey
            ts.sort()
            for j in range(len(ts)-1):
                lo, hi = ts[j], ts[j+1]
                if hi-lo < 1e-6:
                    continue
                mid = (lo+hi) * .5
                z = az+dz*mid
                if not (low + .05 < z < high - .05):
                    continue
                x, y = ax + dx*mid, ay + dy*mid
                if self.inside(x,y,ring) and not any(self.inside(x,y,h) for h in holes):
                    return False
        return True


class Bot:
    name = 'Bizz01-Aegis-v3'

    def __init__(self, hello: mf.Hello):
        self.config = hello['config']
        directory = hello.get('map', {}).get('dir', '/arena')
        self.nav = NavGrid(directory)
        self.sight = MapSight(directory)
        self.rng = random.Random(hello.get('seed', 0))
        self.route_path = []
        self.route_goal = None
        self.route_at = -99
        self.aim_target = None
        self.aim_since = 0
        self.farm_focus = None
        self.last_pos = None
        self.jet_until = -10
        self.last_damage = -100
        self.last_shot = -100
        self.orbit_dir = -1 if hello.get('slot', 0) % 2 else 1
        self.last_seen_enemy = None
        self.last_seen_time = -100
        self.prev_hp = float(self.config['mech'].get('hp', 1400))
        self.enemy_id = None
        self.enemy_locked_at = -99
        self.cooldown_goal = None
        self.wall_avoid_until = -10

    def upgrade(self, obs, me):
        choices = obs.get('offer', {}).get('choices', [])
        hp = me['hp'] / max(1, me['max_hp'])
        kinds = {w['kind'] for w in me['weapons'] if w}
        early = obs['t'] < 120
        score = {
            'nanites': 125 if hp < .90 else 114,
            'armor': 110,
            'servos': 109 if early else 102,
            'plating': 105 if hp < .75 else 99,
            'gun_rate': 103 if 'gun' in kinds else 45,
            'gun_damage': 100 if 'gun' in kinds else 45,
            'gun_mag': 96,
            'laser_damage': 117,
            'laser_charge': 119,
            'laser_pierce': 81,
            'lightning_range': 112,
            'lightning_efficiency': 96,
            'lightning_chain': 79,
            'mortar_reload': 100,
            'mortar_velocity': 84,
            'mortar_splash': 93,
            'jets': 85,
            'capacitor': 80,
            'reactor': 75,
            'gun_velocity': 70,
            'gyros': 70,
            'sensors': 40,
        }
        if hp < .42:
            score['armor'] += 9
            score['plating'] += 7
        if obs['t'] > 220:
            score['servos'] += 4
        return max(range(len(choices)), key=lambda i:score.get(choices[i]['id'], 55)) if choices else 0

    @staticmethod
    def gap(a,b):
        return abs(angle_delta(a,b))

    def aim(self, me, thing, kind):
        """Predict intercept; LOS checked separately for actual final line."""
        pos, vel = thing['pos'], thing.get('vel', [0,0,0])
        x, y, z = me['pos']
        stats = me.get('stats', {})
        mz = z + stats.get('muzzle_height', 7.0)
        target_height = {'mech':4.7, 'npc':1.8, 'crate':1.2}.get(thing['kind'], 3)
        tz = pos[2] + target_height
        w = next((w for w in me['weapons'] if w and w['kind']==kind), None)
        if kind == 'gun':
            speed = w['stats'].get('speed',252) if w else 252
            tt = max(0, math.hypot(pos[0]-x,pos[1]-y)-3) / max(1,speed)
            for _ in range(2):
                px,py=pos[0]+vel[0]*tt,pos[1]+vel[1]*tt
                tt = math.sqrt((px-x)**2 + (py-y)**2 + (tz+vel[2]*tt-mz)**2) / max(1,speed)
            tt = min(tt,1.25)
        elif kind == 'mortar':
            speed = w['stats'].get('speed',72) if w else 72
            gravity = self.config['weapons']['mortar'].get('gravity', 22)
            tt = max(0.1, math.hypot(pos[0]-x,pos[1]-y) / max(30,speed*.78))
            # Repeatedly solve moving-target ballistic equation, using low arc.
            for _ in range(3):
                px,py=pos[0]+vel[0]*tt,pos[1]+vel[1]*tt
                rz = tz + clamp(vel[2],-5,5)*tt - mz
                horiz=math.hypot(px-x,py-y)
                discr = speed**4 - gravity * (gravity*horiz*horiz + 2*rz*speed*speed)
                if discr <= 0 or horiz < 13:
                    return None
                theta = math.atan((speed*speed-math.sqrt(discr)) / (gravity*horiz))
                tt = clamp(horiz / max(20, speed*math.cos(theta)),.1,5)
            dx,dy=pos[0]+vel[0]*tt-x,pos[1]+vel[1]*tt-y
            horiz=math.hypot(dx,dy)
            dz=tz+clamp(vel[2],-5,5)*tt-mz
            discr=speed**4-gravity*(gravity*horiz*horiz+2*dz*speed*speed)
            if discr < 0 or horiz < 13:
                return None
            pitch=math.atan((speed*speed-math.sqrt(discr))/(gravity*horiz))
            return math.atan2(dy,dx),clamp(pitch,-.65,1.25),tt
        else:
            tt = .035 if kind == 'laser' else .07
        px, py = pos[0]+vel[0]*tt, pos[1]+vel[1]*tt
        pz = tz + vel[2]*tt
        return math.atan2(py-y,px-x), clamp(math.atan2(pz-mz,max(1,math.hypot(px-x,py-y))),-.68,1.25),tt

    def visible_to_shoot(self, me, thing):
        pos=thing['pos']
        h={'mech':4.5,'npc':1.75,'crate':1.2}.get(thing['kind'],3)
        a=(me['pos'][0],me['pos'][1],me['pos'][2] + me.get('stats',{}).get('muzzle_height',7))
        b=(pos[0],pos[1],pos[2]+h)
        return self.sight.clear(a,b)

    def zone_target(self, me, zone, t, desired=None):
        here=me['pos']
        nc=zone['next_center']
        nr=zone['next_radius']
        margin=min(40,max(15,nr*.11))
        effective=max(6,nr-margin)
        d=d2(here,nc)
        future_risk=d > effective
        time_left=max(0,zone['shrink_end']-t)
        speed=me['stats'].get('walk_speed',11.7)
        need_s= max(0,d-effective)/max(5,speed*.68)
        urgent=(not me['in_zone'] or (future_risk and (time_left < need_s+13 or nr < 105)))
        if desired is not None and d2(desired,nc)< effective:
            return desired,urgent,future_risk
        # Early: route to next-zone interior. Late: avoid center stacking as long as viable.
        if nr>140:
            cx,cy=nc
            if d < effective*.72:
                # Keep walking rather than sitting still in the center waiting to
                # be hunted. Patrol toward a reachable waypoint for six seconds.
                phase=int(t/6.4)
                angle=0.88+phase*2.39996323 + self.orbit_dir*.6
                px=here[0]+56*math.cos(angle)
                py=here[1]+56*math.sin(angle)
                cd=math.hypot(px-cx,py-cy)
                if cd>effective-5 and cd>1:
                    px=cx+(px-cx)*(effective-5)/cd
                    py=cy+(py-cy)*(effective-5)/cd
                return (px,py),urgent,future_risk
            return (cx,cy),urgent,future_risk
        ang=math.atan2(here[1]-nc[1],here[0]-nc[0])
        target=(nc[0]+max(0, effective*.3)*math.cos(ang),nc[1]+max(0,effective*.3)*math.sin(ang))
        return target,urgent,future_risk

    def choose_objective(self, obs, me, enemies, near, zone_urgent):
        t=obs['t']; here=me['pos']; zone=obs['zone']; nc=zone['next_center']; nr=zone['next_radius']
        energy=me['energy']; hp=me['hp']/max(1,me['max_hp'])
        survival=(hp < .35 or len([e for e in enemies if d2(here,e['pos'])<80])>=3)
        candidates=[]
        owned={w['kind'] for w in me['weapons'] if w}
        def add(score,id,xy,typ,extra=None):
            if nr<60 and d2(xy,nc)>max(4,nr-7): return
            candidates.append((score,id,tuple(xy[:2]),typ,extra))
        # Power cubes: repair by increasing max HP and scale damage.
        for c in obs.get('cubes',[]):
            d=d2(here,c['pos'])
            if d>145:continue
            danger=min((d2(c['pos'],e['pos']) for e in enemies),default=999)
            if danger < 24 and not survival: continue
            score=115-1.08*d + (16 if me['cubes']<6 else 0)
            if danger<60: score -= 36
            add(score,c['id'],c['pos'],'cube')
        # Starting with only a gun means weapon pickup is critical.
        for p in obs.get('pickups',[]):
            kind=p['kind']; d=d2(here,p['pos'])
            if kind in owned or kind=='gun' or d > 155: continue
            danger=min((d2(p['pos'],e['pos']) for e in enemies),default=999)
            if danger<34 and not survival: continue
            priority={'laser':160, 'lightning':135, 'mortar':135}.get(kind,0)
            if me['weapons'][1] is not None:
                priority-=75
            if 'lightning' in owned and kind == 'laser': priority+=20
            if danger<65:priority-=35
            score=priority-0.92*d
            add(score,p['id'],p['pos'],'weapon',kind)
        # Early dedicated farm: each NPC = 2 cubes; each crate = 1 cube + chance at weapon.
        if t<225 and near>52 and not zone_urgent and not survival:
            for npc in obs['mechs']:
                if npc['kind']!='npc':continue
                d=d2(here,npc['pos'])
                if d>158 or abs(npc['pos'][2]-here[2])>25:continue
                danger=min((d2(npc['pos'],e['pos']) for e in enemies),default=999)
                if danger<55:continue
                los=self.visible_to_shoot(me,npc)
                score=116-0.63*d+(18 if npc['hp']<150 else 0)+(14 if los else -20)
                if t>115:score-=16
                add(score,npc['id'],npc['pos'],'npc')
            for crate in obs['crates']:
                d=d2(here,crate['pos'])
                if d>135 or abs(crate['pos'][2]-here[2])>19:continue
                danger=min((d2(crate['pos'],e['pos']) for e in enemies),default=999)
                if danger < 48:continue
                c={'kind':'crate','pos':crate['pos'],'vel':[0,0,0],'hp':crate['hp'],'id':crate['id']}
                los=self.visible_to_shoot(me,c)
                score=89-.63*d+(24 if me['weapons'][1] is None else 0)+(25 if crate['hp']<160 else 0)+(9 if los else -23)
                add(score,crate['id'],crate['pos'],'crate')
        # Full HP and 3 cubes for a drop, if we can get there before/shortly after landing.
        if t>65 and not survival:
            for dp in obs.get('drops',[]):
                d=d2(here,dp['pos'])
                dd=min((d2(dp['pos'],e['pos']) for e in enemies),default=999)
                if dd<47 or d>155: continue
                if d / max(6,me['stats'].get('walk_speed',11.7)) > dp['lands_in']+13:continue
                val=165-0.80*d-(40 if dd<100 else 0)
                if me['hp']>me['max_hp']*.88:val-=20
                add(val,dp['id'],dp['pos'],'drop')
        # If survival is critical, seek cover or zone instead of risky distractions.
        if survival:
            candidates=[c for c in candidates if c[3] in ('cube','weapon','drop') and c[0]>75]
        if candidates:
            candidates.sort(key=lambda c:c[0],reverse=True)
            if self.farm_focus is not None:
                old=next((c for c in candidates if c[1] == self.farm_focus),None)
                if old and old[0] >= candidates[0][0] - 18:
                    return old
            top=candidates[0]
            if top[0]> (30 if t < 130 else 43):
                self.farm_focus=top[1]
                return top
        self.farm_focus=None
        return None

    def choose_target(self, obs, me, enemies, objective, near):
        here=me['pos']; t=obs['t']
        options=[]
        for e in enemies:
            dist=d2(here,e['pos'])
            if dist>255: continue
            los=self.visible_to_shoot(me,e)
            if not los: continue
            hp=e.get('hp',1400)
            hpmax=e.get('max_hp',1400)
            health_ratio=hp/max(1,hpmax)
            closing=(e.get('vel',[0,0,0])[0]*(here[0]-e['pos'][0])
                     +e.get('vel',[0,0,0])[1]*(here[1]-e['pos'][1]))/max(10,dist)
            score=(2.1-0.008*dist + 1.75*(1-health_ratio)
                   +(.65 if dist<75 else 0) +(.35 if closing>3 else 0)
                   +(.33 if e['id']==self.enemy_id else 0))
            options.append((score,e,dist))
        if options:
            options.sort(key=lambda o:o[0],reverse=True)
            _,enemy,d=options[0]
            # Defend against enemies and finish damaged targets. Otherwise obtain
            # a power spike from a nearby NPC or crate before dueling a squad.
            if d<170 or (enemy.get('hp',1400)<650 and d<215) or (near<215 and objective is None):
                self.enemy_id=enemy['id']; self.enemy_locked_at=t
                return enemy
        if objective and objective[3] in ('npc','crate') and near>145:
            for obj in (obs['mechs'] if objective[3]=='npc' else obs['crates']):
                if obj['id'] != objective[1]: continue
                thing=obj
                if objective[3]=='crate':
                    thing={'id':obj['id'],'kind':'crate','pos':obj['pos'],
                           'vel':[0,0,0],'hp':obj['hp']}
                if self.visible_to_shoot(me,thing):return thing
        if options:
            self.enemy_id=options[0][1]['id'];self.enemy_locked_at=t
            return options[0][1]
        # Shoot a reachable crate/NPC while passing, even when no farm target was chosen.
        if near > 135 and t<185:
            candidates=[]
            for obj in obs.get('crates',[]):
                dd=d2(here,obj['pos'])
                if dd<100:
                    c={'id':obj['id'],'kind':'crate','pos':obj['pos'],'vel':[0,0,0],'hp':obj['hp']}
                    if self.visible_to_shoot(me,c):candidates.append((dd+15,c))
            for obj in obs['mechs']:
                if obj['kind']=='npc' and d2(here,obj['pos'])<120 and self.visible_to_shoot(me,obj):
                    candidates.append((d2(here,obj['pos'])-15,obj))
            if candidates: return min(candidates,key=lambda i:i[0])[1]
        return None

    def route_move(self, here, goal, enemies, t):
        if self.route_goal is None or d2(self.route_goal,goal)>21 or t-self.route_at>2.4 or not self.route_path:
            coords=[tuple(e['pos'][:2]) for e in enemies[:4]]
            self.route_path=self.nav.route(here,goal,coords)
            self.route_goal=goal
            self.route_at=t
        while len(self.route_path)>1 and d2(here,self.route_path[0])<9:
            self.route_path.pop(0)
        if self.nav.ok:
            for k in range(min(len(self.route_path)-1,10),0,-1):
                if self.nav.clear(here,self.route_path[k]):
                    self.route_path=self.route_path[k:]
                    break
        dest=self.route_path[0] if self.route_path else goal
        return norm(dest[0]-here[0],dest[1]-here[1]),dest

    def combat_move(self, me, enemies, target, moving, urgent, t):
        x,y=me['pos'][:2]
        near=min((d2((x,y),e['pos']) for e in enemies),default=1000)
        n1=sum(d2((x,y),e['pos'])<100 for e in enemies)
        health=me['hp']/max(1,me['max_hp'])
        if not enemies or near>160:
            return moving
        e=target if target and target['kind']=='mech' else enemies[0]
        dd=d2((x,y),e['pos'])
        to_x,to_y=norm(e['pos'][0]-x,e['pos'][1]-y)
        away_x,away_y=-to_x,-to_y
        side=self.orbit_dir
        # Reverse strafe occasionally, especially when dodging a hitscan charge.
        if int(t/3.7)%2:side=-side
        if dd<32: away=1.6
        elif dd<63: away=.98
        elif dd<105: away=.28
        elif dd<140: away=-.05
        else: away=-.30
        if health<.43 or n1>=3:
            away+=.65
        if urgent: away*=.45
        tangent_x,tangent_y=-to_y*side,to_x*side
        blend=clamp((155-near)/115,0,.82)
        # Maintain motion perpendicular to enemy aiming, damaging while kiting.
        desired_x=away_x*away + tangent_x*(.85 if n1<3 else .55)
        desired_y=away_y*away + tangent_y*(.85 if n1<3 else .55)
        ddx,ddy=norm(desired_x,desired_y)
        return norm(moving[0]*(1-blend)+ddx*blend,moving[1]*(1-blend)+ddy*blend)

    def dodge(self, obs, me, move):
        x,y,z=me['pos']; sx,sy=move
        bullets=obs.get('projectiles',[])
        avoidx,avoidy=0.0,0.0
        risk=0.0
        for b in bullets:
            p=b.get('pos', [0,0,0]);v=b.get('vel',[0,0,0])
            rx,ry=p[0]-x,p[1]-y
            vx,vy=v[0]-me['vel'][0],v[1]-me['vel'][1]
            vv=vx*vx+vy*vy
            if vv<25 or rx*rx+ry*ry>280**2: continue
            intercept=-(rx*vx+ry*vy)/vv
            if intercept<.015 or intercept>.75:continue
            px,py=rx+vx*intercept,ry+vy*intercept
            miss=math.hypot(px,py)
            spread=11 if b.get('kind')=='mortar' else 5.4
            bz=p[2]+v[2]*intercept
            if abs(bz-(z+4.7))>(12 if b.get('kind')=='mortar' else 8):continue
            if miss>spread:continue
            nx,ny=norm(-vy,vx)
            sign=-1 if px*nx+py*ny>0 else 1
            power=(spread-miss)/spread * (1-.35*intercept)
            avoidx+=nx*sign*power;avoidy+=ny*sign*power
            risk+=power
        # A laser charges toward your current position; immediate lateral motion matters.
        for e in obs['mechs']:
            if e['kind']!='mech' or not e.get('charging'):continue
            d=d2(me['pos'],e['pos'])
            if d>275:continue
            facing=math.atan2(y-e['pos'][1],x-e['pos'][0])
            err=angle_delta(e.get('aim',[0,0])[0],facing)
            if abs(err)<.12:
                to_x,to_y=norm(x-e['pos'][0],y-e['pos'][1])
                side=1 if err>0 else -1
                avoidx+=-to_y*side*.7;avoidy+=to_x*side*.7
                risk+=.45
        if risk>0:
            ax,ay=norm(avoidx,avoidy)
            factor=clamp(risk*.95,0,.87)
            sx,sy=norm(sx*(1-factor)+ax*factor,sy*(1-factor)+ay*factor)
        return (sx,sy),risk

    def fight(self, obs, me, target, act, enemies, near, urgent):
        w0,w1=me['weapons'];t=obs['t']
        if target is None:
            # Track approaching attackers; otherwise sweep the horizon to see loot.
            if enemies:
                e=enemies[0]
                yaw=math.atan2(e['pos'][1]-me['pos'][1],e['pos'][0]-me['pos'][0])
            else:
                # Continuous full-circle scanning sees loot and hunters in all
                # directions, not only a limited arc around our move heading.
                yaw=angle_delta(t*.94 + self.orbit_dir*.7,0.0)
            act['aim']=[round(yaw,5),0.0]
            if w1 and w1['kind']=='laser' and w1.get('charge',0)>.04:
                act['fire'][1]=False
            if w0 and w0['kind']=='gun' and w0.get('ammo',12)<w0['stats'].get('mag',12):
                act['reload']=True
            return
        kind=target['kind']
        dist=d2(me['pos'],target['pos'])
        if target['id']!=self.aim_target:
            self.aim_target=target['id'];self.aim_since=t
        gun=self.aim(me,target,'gun') if w0 and w0['kind']=='gun' else None
        sec=w1['kind'] if w1 else None
        secstats=w1.get('stats',{}) if w1 else {}
        cansec=(sec and self.visible_to_shoot(me,target) and
                dist<secstats.get('range',300)-2)
        shot=gun
        active='gun'
        secfire=False
        if sec=='laser' and cansec:
            charge=w1.get('charge',0)
            total=w1['stats'].get('charge',.75)
            cd=w1.get('cooldown',0)
            if cd<.13 or charge>.02:
                laser=self.aim(me,target,'laser')
                # Keep gun pressure through the laser's charge period. Switch to
                # exact hitscan aim just before releasing the charge.
                if charge>=.64*total or not gun or dist>170:
                    shot=laser;active='laser'
                if charge < .98*total:
                    secfire=True
                elif charge>=.98*total:
                    if laser:
                        tolerance=clamp(math.atan2(1.7,max(15,dist)),.008,.070)
                        aligned=(self.gap(laser[0],me['aim'][0])<tolerance and
                                 abs(laser[1]-me['aim'][1])<max(.016,tolerance*1.4))
                        secfire=not aligned and charge<total*1.45
                    else:
                        secfire=False
                if charge>total*1.6:secfire=False
        elif sec=='mortar' and cansec and 23<dist<215 and w1.get('cooldown',0)<.12:
            mortar=self.aim(me,target,'mortar')
            if mortar:
                shot=mortar;active='mortar'
        elif sec=='lightning' and cansec and dist<secstats.get('range',45)-1:
            # Gun remains active so lightning and gun can be fired together.
            if gun:shot=gun;active='gun'
            else:
                shot=self.aim(me,target,'lightning');active='lightning'
        if shot is not None:
            yaw,pitch,_=shot
            act['aim']=[round(yaw,6),round(pitch,6)]
        else:
            act['aim']=[math.atan2(target['pos'][1]-me['pos'][1],target['pos'][0]-me['pos'][0]),0]
        if gun:
            # The server turns the torso *before* shooting this decision; a
            # small current-aim error can be corrected within the next tick.
            accuracy=max(.065, min(.105, math.atan2(2.2,max(12,dist))+.060))
            yawgap=self.gap(gun[0],me['aim'][0])
            pitchgap=abs(gun[1]-me['aim'][1])
            # Important: don't fire inaccurate rounds just because a target is visible.
            if (dist<w0['stats'].get('range',230)-4 and yawgap<accuracy and
                    pitchgap<max(.060,accuracy*1.12) and w0.get('ammo',0)>0):
                act['fire'][0]=True
            if w0.get('ammo',12)<=0 or (near>110 and w0.get('ammo',12)<w0['stats'].get('mag',12)*.45):
                act['reload']=True
        if not w1 or not cansec:return
        if sec=='laser':
            act['fire'][1]=bool(secfire)
        elif sec=='lightning':
            cost=secstats.get('energy',3)
            aim=self.aim(me,target,'lightning')
            if (aim and dist < secstats.get('range',45)-3 and
                    self.gap(aim[0],me['aim'][0])<.17 and me['energy']>cost+7):
                act['fire'][1]=True
        elif sec=='mortar' and active=='mortar' and shot:
            # Mortar splash harms self inside 20m; don't lob shells at own feet.
            yawgap=self.gap(shot[0],me['aim'][0]);pitchgap=abs(shot[1]-me['aim'][1])
            acc=clamp(math.atan2(5,max(20,dist)),.02,.085)
            if yawgap<acc and pitchgap<.052 and dist>27:
                act['fire'][1]=True

    def act(self, obs: mf.Observation) -> mf.Action:
        me=obs['self'];t=obs['t'];x,y,z=me['pos'];here=(x,y)
        hp=me['hp']/max(1,me['max_hp'])
        enemies=sorted((m for m in obs['mechs'] if m['kind']=='mech'),key=lambda e:d2(here,e['pos']))
        near=d2(here,enemies[0]['pos']) if enemies else 1e6
        zone=obs['zone']
        action={'tick':obs['tick'],'fire':[False,False]}
        if obs.get('offer'):
            action['pick']=self.upgrade(obs,me)
        owned={w['kind'] for w in me['weapons'] if w}
        # Loot pickup is independent of moving and firing.
        radius=self.config.get('loot',{}).get('pickup_radius',4)
        for p in sorted(obs.get('pickups',[]),key=lambda p:d2(here,p['pos'])):
            if (p['kind'] in owned or p['kind']=='gun' or
                    d2(here,p['pos'])>radius-.4 or abs(p['pos'][2]-z)>3.5):continue
            if me['weapons'][1] is None:
                action['take']=p['id']
                break
            # Replace mortar/lightning with laser, unless enemy is very close.
            if (p['kind']=='laser' and me['weapons'][1]['kind'] in ('mortar','lightning') and
                    near>75 and (me['weapons'][1]['kind']!='laser')):
                action['take']=p['id'];action['drop']=1;break
        if me['hp']<self.prev_hp-1:
            self.last_damage=t
        self.prev_hp=me['hp']
        nc=zone['next_center'];nr=zone['next_radius']
        zonegoal,urgent,outer=self.zone_target(me,zone,t)
        objective=self.choose_objective(obs,me,enemies,near,urgent)
        goal=zonegoal;goal_type='zone'
        if objective and not urgent:
            goal=objective[2];goal_type=objective[3]
        if urgent:
            goal=zonegoal;goal_type='zone'
        # If enemies close and threatening, keep distance and shoot them while
        # moving toward a safe waypoint. Never be trapped in a stationary duel.
        close=[e for e in enemies if d2(here,e['pos'])<95]
        if (len(close)>=2 or (hp<.44 and close)) and not urgent:
            fx,fy=0.,0.
            for e in enemies:
                dist=d2(here,e['pos'])
                if dist<155:
                    dx,dy=norm(x-e['pos'][0],y-e['pos'][1])
                    strength=(155-dist)/max(15,dist)
                    fx+=dx*strength;fy+=dy*strength
            dx,dy=norm(fx,fy)
            if dx or dy:
                repel=(x+dx*(72 if hp>.4 else 105),y+dy*(72 if hp>.4 else 105))
                limit=max(6,nr-min(30,nr*.15))
                dd=d2(repel,nc)
                if dd>limit and dd>0:
                    repel=(nc[0]+(repel[0]-nc[0])*limit/dd,nc[1]+(repel[1]-nc[1])*limit/dd)
                if d2(here,repel)>15:
                    goal=repel;goal_type='disengage'
        moving,dest=self.route_move(here,goal,enemies,t)
        target=self.choose_target(obs,me,enemies,objective,near)
        moving=self.combat_move(me,enemies,target,moving,urgent,t)
        moving,hit_risk=self.dodge(obs,me,moving)
        if self.last_pos is None:self.last_pos=(x,y,t)
        elif t-self.last_pos[2]>=1.3:
            if d2(here,self.last_pos)<2.7 and d2(goal,here)>16:
                self.jet_until=t+2.6
                self.route_at=-99
            self.last_pos=(x,y,t)
        # Near an objective, slow enough for the cube/weapon pickup action to work.
        goal_dist=d2(here,goal)
        if (goal_type in ('cube','weapon','drop') and goal_dist<3.2):
            moving=(0,0)
        elif goal_type in ('npc','crate') and goal_dist<8:
            # Get close enough to recover cubes when the target dies.
            moving=(moving[0]*.55,moving[1]*.55)
        energy=me['energy']
        endangered=(near<68 or len(close)>=2 or hit_risk>.45 or hp<.38)
        boost=(energy>36 and ((urgent and goal_dist>24) or
                (endangered and (near<55 and hp<.73 or len(close)>=3 or hit_risk>.65)) or
                (goal_dist>100 and goal_type in ('weapon','cube','drop') and energy>88)))
        needs_jet=t<self.jet_until and energy>19
        action['move']=[round(moving[0],5),round(moving[1],5)]
        action['boost']=bool(boost or (needs_jet and not me['grounded']))
        if me['grounded'] and (needs_jet or
                (near<115 and energy>45 and int(t*1.8)%9==0 and
                 (hit_risk>.1 or hp<.65))):
            action['jump']=True
        self.fight(obs,me,target,action,enemies,near,urgent)
        return action


mf.run(Bot)
