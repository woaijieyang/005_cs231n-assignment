$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentName = 'cs231n_zx_test'
$environmentInfo = conda env list --json | ConvertFrom-Json
$existing = @($environmentInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $environmentName })
if ($existing.Count -eq 0) {
    conda create -n $environmentName python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
    $environmentInfo = conda env list --json | ConvertFrom-Json
    $existing = @($environmentInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $environmentName })
}
if ($existing.Count -ne 1) { throw 'Expected exactly one cs231n_zx_test environment' }
$pythonExe = Join-Path $existing[0] 'python.exe'
& $pythonExe -c "import sys; assert sys.version_info[:2] == (3,11), 'Existing environment must use Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw 'Existing environment incompatible; no changes applied' }
& $pythonExe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $pythonExe -m pip install -r (Join-Path $projectRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $pythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $pythonExe -m ipykernel install --user --name $environmentName --display-name $environmentName
if ($LASTEXITCODE -ne 0) { throw 'Kernel registration failed' }
foreach ($assignment in @('02_Assignment2', '03_Assignment3')) {
    Push-Location (Join-Path $projectRoot "$assignment\cs231n")
    try {
        if (Get-Command gcc -ErrorAction SilentlyContinue) {
            & $pythonExe setup.py build_ext --inplace --compiler=mingw32 -D MS_WIN64
        } else {
            & $pythonExe setup.py build_ext --inplace
        }
        if ($LASTEXITCODE -ne 0) { throw "Cython build failed: $assignment" }
    } finally { Pop-Location }
}
& $pythonExe -c "import torch; assert torch.cuda.is_available(); x=torch.randn(4,4,device='cuda',requires_grad=True); x.square().sum().backward(); print(torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA execution check failed' }
Write-Output 'Environment and Jupyter kernel ready: cs231n_zx_test'
