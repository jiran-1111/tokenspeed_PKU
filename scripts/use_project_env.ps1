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

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EnvironmentPrefix)) {
    $EnvironmentPrefix = Join-Path $ProjectRoot '.conda-env'
}
$EnvironmentPrefix = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPrefix)
$EnvironmentPrefixRoot = [System.IO.Path]::GetPathRoot($EnvironmentPrefix)
if ($EnvironmentPrefix.Length -gt $EnvironmentPrefixRoot.Length) {
    $EnvironmentPrefix = $EnvironmentPrefix.TrimEnd([char[]]@('\', '/'))
}
if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPrefix 'conda-meta') -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $EnvironmentPrefix 'python.exe') -PathType Leaf)) {
    throw 'Project Conda environment is missing. Run scripts/setup_env.ps1 first, or supply -EnvironmentPrefix for an existing environment.'
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'Conda is required. Initialize Conda in PowerShell before activating the project environment.'
}

$env:CONDA_ENVS_PATH = Join-Path $ProjectRoot '.conda-envs'
$env:CONDA_PKGS_DIRS = Join-Path $ProjectRoot '.conda-pkgs'
$env:CONDA_REGISTER_ENVS = 'false'
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache/pip'
$env:HF_HOME = Join-Path $ProjectRoot '.cache/huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache/torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $ProjectRoot '.cache/torch-extensions'
$env:TRITON_HOME = Join-Path $ProjectRoot '.cache/triton'
$env:TRITON_CACHE_DIR = Join-Path $env:TRITON_HOME 'cache'
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot '.cache'
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache/matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $ProjectRoot '.cache/numba'
$env:ASCEND_WORK_PATH = Join-Path $ProjectRoot '.tmp/ascend_work'
$env:TMPDIR = Join-Path $ProjectRoot '.tmp'
$env:TEMP = $env:TMPDIR
$env:TMP = $env:TMPDIR

$ProjectDirectories = @(
    $env:CONDA_ENVS_PATH
    $env:CONDA_PKGS_DIRS
    $env:PIP_CACHE_DIR
    $env:HF_HOME
    $env:TORCH_HOME
    $env:TORCH_EXTENSIONS_DIR
    $env:TRITON_CACHE_DIR
    $env:MPLCONFIGDIR
    $env:NUMBA_CACHE_DIR
    $env:ASCEND_WORK_PATH
    $env:TMPDIR
    (Join-Path $ProjectRoot 'models')
    (Join-Path $ProjectRoot 'data')
    (Join-Path $ProjectRoot 'results/raw')
    (Join-Path $ProjectRoot 'results/plots')
)
foreach ($ProjectDirectory in $ProjectDirectories) {
    New-Item -ItemType Directory -Force -Path $ProjectDirectory -ErrorAction Stop | Out-Null
}
$ProjectPythonPaths = @(
    (Join-Path $ProjectRoot 'python')
    (Join-Path $ProjectRoot 'tokenspeed-kernel/python')
    (Join-Path $ProjectRoot 'tokenspeed-kernel-npu/python')
)
if ($env:PYTHONPATH) {
    $ProjectPythonPaths += $env:PYTHONPATH
}
$env:PYTHONPATH = $ProjectPythonPaths -join [System.IO.Path]::PathSeparator

$global:LASTEXITCODE = 0
$CondaHook = & conda 'shell.powershell' 'hook'
if ($LASTEXITCODE -ne 0) {
    throw 'Conda could not initialize its PowerShell hook.'
}
$CondaHook | Out-String | Invoke-Expression -ErrorAction Stop
conda activate $EnvironmentPrefix
$ActivatedPrefix = $env:CONDA_PREFIX
if (-not [string]::IsNullOrWhiteSpace($ActivatedPrefix)) {
    $ActivatedPrefix = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ActivatedPrefix)
    $ActivatedPrefixRoot = [System.IO.Path]::GetPathRoot($ActivatedPrefix)
    if ($ActivatedPrefix.Length -gt $ActivatedPrefixRoot.Length) {
        $ActivatedPrefix = $ActivatedPrefix.TrimEnd([char[]]@('\', '/'))
    }
}
if ($LASTEXITCODE -ne 0 -or
    -not [string]::Equals($ActivatedPrefix, $EnvironmentPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Conda did not activate the requested project environment.'
}
