>output.tmp (dir /b /a-d | find /v /c "")

<output.tmp (
  set /p line1=
)
echo %line1%

for /l %%I in (1,1,%line1%) do (
echo %%I
)
