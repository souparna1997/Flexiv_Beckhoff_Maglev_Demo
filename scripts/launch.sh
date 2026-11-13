#!/bin/sh
# Load the demo XML through the native OpenSai controller.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
config=${1:-xml_config_files/rizon_real.xml}
if [ "$#" -gt 1 ]; then
    echo "Usage: sh scripts/launch.sh [xml_config_files/rizon_real.xml]" >&2
    exit 1
fi
case "$config" in
    rizon_real.xml|xml_config_files/rizon_real.xml|"$project_root/xml_config_files/rizon_real.xml") ;;
    *) echo "This launcher supports xml_config_files/rizon_real.xml" >&2; exit 1 ;;
esac
if [ ! -f "$project_root/xml_config_files/rizon_real.xml" ]; then
    echo "Missing xml_config_files/rizon_real.xml" >&2
    exit 1
fi
if [ ! -x "$project_root/build/maglev_opensai" ]; then
    echo "Build the controller first:" >&2
    echo "  cmake -S \"$project_root\" -B \"$project_root/build\"" >&2
    echo "  cmake --build \"$project_root/build\" -j2" >&2
    exit 1
fi
cd "$project_root"
echo "Starting the OpenSai controller with rizon_real.xml; it holds the measured joint pose."
echo "Run the pickup sequence from another terminal after controller startup:"
echo "  python3 \"$project_root/rizon_pick_place.py\""
exec "$project_root/build/maglev_opensai"
