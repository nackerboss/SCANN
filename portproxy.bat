@echo off

:: -----------------------------------------------------
:: 1. Get the current WSL2 IP Address
:: -----------------------------------------------------
:: The FOR /F command captures the output of 'wsl hostname -I'
:: and stores the IP (the first token) in the variable %%i.
FOR /F "tokens=1" %%i IN ('wsl hostname -I') DO (
    SET "WSL_IP=%%i"
)

:: Check if the IP was successfully retrieved (simple check)
IF "%WSL_IP%"=="" (
    ECHO Error: Could not retrieve WSL IP address. Is WSL running?
    GOTO :EOF
)

ECHO Found WSL IP: %WSL_IP%

:: -----------------------------------------------------
:: 2. Add the Port Forwarding Rule (Port 8000)
:: -----------------------------------------------------
ECHO Adding port proxy rule for 0.0.0.0:8000 ^> %WSL_IP%:8000
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=%WSL_IP%

ECHO.
ECHO Port Proxy Rules After Adding:
netsh interface portproxy show v4tov4

:: -----------------------------------------------------
:: 3. Pause and Cleanup
:: -----------------------------------------------------
ECHO.
ECHO The forwarding rule is now active.
ECHO Press any key to DELETE the port proxy rule and EXIT.
pause > nul

ECHO Deleting port proxy rule...
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8000

ECHO.
ECHO Port Proxy Rules After Deleting:
netsh interface portproxy show v4tov4

ECHO Done.