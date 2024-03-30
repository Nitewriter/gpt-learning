[CmdletBinding()]
param (
    [Parameter()]
    [string]
    $Command = "deps"
)

# Check if OS is Windows
if ($env:OS -ne "Windows_NT") {
    Write-Host "This script is only for Windows"
    exit
}

# Variables
$VENV = ".venv"

function CreateEnv {
    if (-not (Test-Path $VENV)) {
        python -m venv $VENV
    }
}

function InstallDependencies {
    # Activate virtual environment
    . $VENV\Scripts\Activate.ps1

    python -m pip install --upgrade pip
    python -m pip install poetry
    poetry install
}

# Switch over provided command
switch ($Command) {
    "venv" {
        CreateEnv
        break
    }
    "deps" {
        CreateEnv
        InstallDependencies
        break
    }
    "clean" {
        if (Test-Path $VENV) {
            . $VENV\Scripts\Deactivate.ps1 2>$null
            Remove-Item -Recurse -Force $VENV
            Write-Host "Virtual environment removed successfully"
        }
        else {
            Write-Host "Virtual environment does not exist"
        }
    }
    default {
        Write-Host "Invalid command ($Command): Supported commands are 'venv', 'deps', 'clean'"
    }
}