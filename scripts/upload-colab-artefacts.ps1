# Step 6 - after Colab finishes, place the 5 downloaded files under
#   models\cards-colab\v1\
# then run this script from the project root:
#   .\scripts\upload-colab-artefacts.ps1
#
# It mirrors the artefacts to S3 and also copies the two figures into
# report\figures\ so they're ready to drop into the report.

param(
    [string]$Version = "v1",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvFile)) {
    throw "Missing $EnvFile - create it with AWS_PROFILE, AWS_REGION, S3_BUCKET"
}
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*$') {
        Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2]
    }
}
foreach ($k in @("AWS_PROFILE", "AWS_REGION", "S3_BUCKET")) {
    if (-not (Get-Item ("Env:" + $k) -ErrorAction SilentlyContinue)) {
        throw "Missing $k in $EnvFile"
    }
}

$Src = "models\cards-colab\$Version"
if (-not (Test-Path $Src)) {
    throw "Source directory '$Src' not found. Put the Colab downloads there first."
}

$Required = @(
    "model.tar.gz",
    "training_curves.png",
    "confusion_matrix.png",
    "metadata.json",
    "classification_report.json"
)
foreach ($f in $Required) {
    if (-not (Test-Path (Join-Path $Src $f))) {
        throw "Missing artefact: $Src\$f"
    }
}

Write-Host "profile : $env:AWS_PROFILE"
Write-Host "region  : $env:AWS_REGION"
Write-Host "bucket  : $env:S3_BUCKET"
Write-Host "source  : $Src"
Write-Host ""
aws sts get-caller-identity --profile $env:AWS_PROFILE | Out-Host

$Prefix = "models/cards-colab/$Version"
$LogsPrefix = "logs/cards-colab/$Version"

Write-Host "`n-- uploading model artefact --"
aws s3 cp "$Src\model.tar.gz" "s3://$env:S3_BUCKET/$Prefix/model.tar.gz" --profile $env:AWS_PROFILE
aws s3 cp "$Src\metadata.json" "s3://$env:S3_BUCKET/$Prefix/metadata.json" --profile $env:AWS_PROFILE
aws s3 cp "$Src\classification_report.json" "s3://$env:S3_BUCKET/$Prefix/classification_report.json" --profile $env:AWS_PROFILE

Write-Host "`n-- uploading figures --"
aws s3 cp "$Src\training_curves.png" "s3://$env:S3_BUCKET/$LogsPrefix/training_curves.png" --profile $env:AWS_PROFILE
aws s3 cp "$Src\confusion_matrix.png" "s3://$env:S3_BUCKET/$LogsPrefix/confusion_matrix.png" --profile $env:AWS_PROFILE

Write-Host "`n-- copying figures into report\figures\ --"
$FigDir = "report\figures"
New-Item -ItemType Directory -Force -Path $FigDir | Out-Null
Copy-Item "$Src\training_curves.png"  "$FigDir\cards_training_curves.png"  -Force
Copy-Item "$Src\confusion_matrix.png" "$FigDir\cards_confusion_matrix.png" -Force

Write-Host "`n-- S3 listing --"
aws s3 ls "s3://$env:S3_BUCKET/$Prefix/" --profile $env:AWS_PROFILE --human-readable --summarize | Out-Host
aws s3 ls "s3://$env:S3_BUCKET/$LogsPrefix/" --profile $env:AWS_PROFILE --human-readable --summarize | Out-Host

Write-Host "`nDone. Next: run the SageMaker driver to repeat training on AWS."
