# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

[CmdletBinding()]
param(
    [string]$EnvironmentPrefix
)

# Native Windows supports the hardware-independent course tests.
# Ascend inference is installed separately with setup_env.sh on Linux.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EnvironmentPrefix)) {
    $EnvironmentPrefix = Join-Path $ProjectRoot '.conda-env'
}
$EnvironmentPrefix = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPrefix)
$EnvironmentPrefixRoot = [System.IO.Path]::GetPathRoot($EnvironmentPrefix)
if ($EnvironmentPrefix.Length -gt $EnvironmentPrefixRoot.Length) {
    $EnvironmentPrefix = $EnvironmentPrefix.TrimEnd([char[]]@('\', '/'))
}
$CourseRequirements = Join-Path $ProjectRoot 'requirements/course-test.txt'
if (-not (Test-Path -LiteralPath $CourseRequirements -PathType Leaf)) {
    throw 'The course test requirements file is missing from this checkout.'
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'Conda is required. Initialize Conda in PowerShell before running setup.'
}

$env:CONDA_ENVS_PATH = Join-Path $ProjectRoot '.conda-envs'
$env:CONDA_PKGS_DIRS = Join-Path $ProjectRoot '.conda-pkgs'
$env:CONDA_REGISTER_ENVS = 'false'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache/pip'
$env:TMPDIR = Join-Path $ProjectRoot '.tmp'
$env:TEMP = $env:TMPDIR
$env:TMP = $env:TMPDIR
foreach ($ProjectDirectory in @(
    $env:CONDA_ENVS_PATH
    $env:CONDA_PKGS_DIRS
    $env:PIP_CACHE_DIR
    $env:TMPDIR
)) {
    New-Item -ItemType Directory -Force -Path $ProjectDirectory -ErrorAction Stop | Out-Null
}

if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPrefix 'conda-meta') -PathType Container)) {
    $global:LASTEXITCODE = 0
    & conda create --prefix $EnvironmentPrefix --yes python=3.11 pip
    if ($LASTEXITCODE -ne 0) {
        throw 'Conda could not create the project environment.'
    }
}
. (Join-Path $PSScriptRoot 'use_project_env.ps1') -EnvironmentPrefix $EnvironmentPrefix
& (Join-Path $EnvironmentPrefix 'python.exe') -m pip install --requirement $CourseRequirements
if ($LASTEXITCODE -ne 0) {
    throw 'The course test dependencies could not be installed.'
}
Write-Output "CPU course environment ready: $EnvironmentPrefix"
Write-Output 'Activate with: . ./scripts/use_project_env.ps1'
