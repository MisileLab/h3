update-packwiz:
	go install github.com/packwiz/packwiz@latest
	clear
	echo "Packwiz has been Updated"
export:
	./simple-packwiz-wrapper.sh export
update:
	./simple-packwiz-wrapper.sh update
refresh:
	./simple-packwiz-wrapper.sh refresh
