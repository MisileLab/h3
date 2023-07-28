ffmpeg -i $1 -c:v h264_nvenc -rc constqp -qp 28 $2
