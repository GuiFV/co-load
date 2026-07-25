' Launches coload-serve.cmd with no visible console window.
' Used by the "coload" scheduled task registered at user logon.
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
cmd = fso.GetParentFolderName(WScript.ScriptFullName) & "\coload-serve.cmd"
shell.Run """" & cmd & """", 0, False
