param(
    [string]$OssExe = "D:\github\Hydrogen-leak\tools\oss.exe",
    [string]$Bundle = "D:\github\Hydrogen-leak\upload_tmp\cfd48_clean_package_20260412.tar.gz",
    [string]$OssTarget = "oss://"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $OssExe)) {
    throw "OSS executable not found: $OssExe"
}
if (-not (Test-Path -LiteralPath $Bundle)) {
    throw "Bundle not found: $Bundle"
}

Write-Host "Checking OSS login..."
& $OssExe ls -s $OssTarget -limit=5 | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "OSS ls failed. Run '$OssExe login' manually first, then rerun this script."
}

Write-Host "Uploading $Bundle to $OssTarget ..."
& $OssExe cp $Bundle $OssTarget -f
if ($LASTEXITCODE -ne 0) {
    throw "OSS upload failed."
}

Write-Host "Upload completed."
Write-Host "Server-side download command:"
Write-Host "  oss cp oss://cfd48_clean_package_20260412.tar.gz /hy-tmp/"
Write-Host "  cd /hy-tmp && tar -xzf cfd48_clean_package_20260412.tar.gz"
