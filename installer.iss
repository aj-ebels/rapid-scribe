; Inno Setup script for Blue Bridge Meeting Companion.
; Prereq: Build the app first with: pyinstaller meetings.spec
; Then open this file in Inno Setup and Compile (or: iscc installer.iss from command line).
; Output: installer_output\Meetings-Setup-1.0.exe (or similar)

#define AppName "Blue Bridge Meeting Companion"
#define AppExe "Blue Bridge Meeting Companion.exe"
#define BuildOutput "dist\Blue Bridge Meeting Companion"

[Setup]
AppId={{B8E92A1C-5D4F-4E2A-9C3B-1D7E8F0A2B3C}
AppName={#AppName}
AppVersion=2.7
AppPublisher=Blue Bridge Solutions
DefaultDirName={autopf}\Blue Bridge Meeting Companion
DefaultGroupName=Blue Bridge Meeting Companion
OutputDir=installer_output
OutputBaseFilename=Blue Bridge Meeting Companion v2.7-Setup
Compression=lzma2
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
; Register uninstall in Settings > Apps (Add or remove programs)
UninstallDisplayName={#AppName}
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "{#BuildOutput}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"
