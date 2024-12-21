#!/bin/bash

check_python() {
	command -v python3 >/dev/null 2>&1 || {
		echo "Python 3 is required but it's not installed. Please install Python 3 and try again. Aborting."
		exit 1
	}
}

check_and_install_opencv() {
	python3 -c "import cv2" 2>/dev/null || {
		echo "OpenCV is not installed. Installing OpenCV..."
		pip install opencv-python
	}
}

setup_environment() {
	echo "Setting up environment..."
}

check_dependencies() {
	echo "Checking if all dependencies in requirements.txt are installed..."

	installed_packages=$(pip freeze | cut -d'=' -f1)

	while IFS= read -r package; do
		[[ "$package" =~ ^#.*$ || -z "$package" ]] && continue

		package_name=$(echo "$package" | sed 's/[><=].*//')

		if echo "$installed_packages" | grep -i -q "^$package_name$"; then
			echo "$package_name is already installed."
		else
			echo "$package_name is missing. Installing..."
			pip install "$package_name"
		fi
	done <requirements.txt
}

main() {
	check_python

	check_and_install_opencv

	setup_environment

	check_dependencies

	python3 src/start.py "$@"
}

main "$@"
