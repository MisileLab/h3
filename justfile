update-packwiz:
	go install github.com/packwiz/packwiz@latest
	go install github.com/Merith-TK/packwiz-wrapper/cmd/pw@latest
	clear
	echo "Packwiz has been Updated"
export:
	pw -b -d modpacks mr export
update:
	pw -b -d modpacks update --all
refresh:
	pw -b -d modpacks refresh
