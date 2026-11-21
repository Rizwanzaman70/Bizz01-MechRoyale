param(
  [string]$KitArchive = ""
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $KitArchive) {
  $Options = @(
    (Join-Path $Root 'mechfight-kit.tar.gz'),
    (Join-Path (Split-Path -Parent $Root) 'mechfight-kit.tar.gz'),
    (Join-Path $env:USERPROFILE 'Downloads\mechfight-kit.tar.gz')
  )
  foreach ($Candidate in $Options) { if (Test-Path $Candidate) { $KitArchive = $Candidate; break } }
}
if (-not $KitArchive -or -not (Test-Path $KitArchive)) {
  throw 'Missing mechfight-kit.tar.gz. Download it from Builders and place it beside this folder, or pass -KitArchive "C:\path\mechfight-kit.tar.gz".'
}
$TempDir = Join-Path $env:TEMP 'Bizz01-MechRoyale-Check'
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
Write-Host "Extracting competition test kit..." -ForegroundColor Cyan
tar -xzf $KitArchive -C $TempDir
if ($LASTEXITCODE -ne 0) { throw 'Could not unpack kit.' }
$KitRoot = Join-Path $TempDir 'mechfight-kit'
$Checker = Join-Path $KitRoot 'check\check.py'
$ArenaPath = (Resolve-Path (Join-Path $KitRoot 'arena')).Path
Write-Host "\n[1/3] Native Python conformance check..." -ForegroundColor Cyan
python $Checker -- python -u (Join-Path $Root 'example_bot.py')
if ($LASTEXITCODE -ne 0) { throw 'Native conformance failed.' }
Write-Host "\n[2/3] Docker build..." -ForegroundColor Cyan
docker build -t bizz01-mechroyale $Root
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed. Ensure Docker Desktop is running.' }
Write-Host "\n[3/3] Competition-equivalent sandbox check..." -ForegroundColor Cyan
python $Checker --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${ArenaPath}:/arena:ro" bizz01-mechroyale
if ($LASTEXITCODE -ne 0) { throw 'Docker sandbox conformance failed.' }
Write-Host "\nAll checks passed. Push this folder to a public GitHub repository." -ForegroundColor Green
