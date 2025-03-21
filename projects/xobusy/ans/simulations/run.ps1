param($input_mp4)

echo $input_mp4
ffmpeg -hwaccel auto -i $input_mp4 -filter_complex "setpts=0.5*PTS" -c:v h264_amf -preset fast output.mp4