Write-Host "Stopping containers..."
docker compose down -v

Write-Host "Starting containers..."
docker compose up -d

Write-Host "Waiting for Postgres..."
for ($i = 0; $i -lt 90; $i++) {
    docker compose exec -T db pg_isready -U user -d toolDB | Out-Null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
}
if ($LASTEXITCODE -ne 0) { throw "Postgres did not become ready." }

Write-Host "Running main.py..."
python .\main.py

Write-Host "Opening psql..."
docker compose exec db psql -U user -d toolDB
