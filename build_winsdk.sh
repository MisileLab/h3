cargo install xwin --locked
export WINSDK_PATH="$HOME/winsdk"
xwin --accept-license splat --preserve-ms-arch-notation --output "$WINSDK_PATH"
clang-cl  --target=x86_64-pc-windows-msvc -fuse-ld=lld /winsdkdir "$WINSDK_PATH/sdk" /vctoolsdir "$WINSDK_PATH/crt" /MD main.c -o main.exe
rm -rfv .xwin*
rm -rfv .xwin-cache
wine main.exe
