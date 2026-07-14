; Inno Setup script for Rapid Scribe.
; Prereq: Build the app first with: pyinstaller meetings.spec
; Then open this file in Inno Setup and Compile (or: iscc installer.iss from command line).
; Output: installer_output\Rapid-Scribe-Setup.exe (or similar)

#define AppName "Rapid Scribe"
#define AppExe "Rapid Scribe.exe"
#define BuildOutput "dist\Rapid Scribe"

[Setup]
AppId={{B8E92A1C-5D4F-4E2A-9C3B-1D7E8F0A2B3C}
AppName={#AppName}
AppVersion=3.18
AppPublisher=Rapid Scribe
DefaultDirName={autopf}\Rapid Scribe
DefaultGroupName=Rapid Scribe
OutputDir=installer_output
OutputBaseFilename=Rapid Scribe v3.18-Setup
Compression=lzma2
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
; Register uninstall in Settings > Apps (Add or remove programs)
UninstallDisplayName={#AppName}
PrivilegesRequired=admin
; x64compatible: native x64 and Windows 11 on ARM (x64 emulation). Plain x64 rejects ARM
; users with the misleading "version of Windows" error (Inno maps arch failures to that text).
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; App dependencies (Python/onnxruntime/WASAPI loopback) require Windows 10+.
MinVersion=10.0

[Messages]
WindowsVersionNotSupported=This installer requires 64-bit Windows 10 or Windows 11 on an Intel or AMD (x64) PC.%n%nSnapdragon / ARM Copilot+ PCs are not supported.%n%nIf you are on a supported PC and still see this, right-click the installer → Properties → Compatibility → uncheck "Run in compatibility mode".

[Code]
function InitializeSetup(): Boolean;
begin
  if IsArm64 then
  begin
    SuppressibleMsgBox(
      'Rapid Scribe is not supported on ARM-based Windows PCs (Snapdragon X, Copilot+ PCs, etc.).' + #13#10 + #13#10 +
      'The app is built for Intel and AMD x64 processors only. Please install on a standard x64 Windows 10 or 11 PC.',
      mbError, MB_OK, MB_OK);
    Result := False;
    Exit;
  end;
  Result := True;
end;

[Files]
Source: "{#BuildOutput}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"
