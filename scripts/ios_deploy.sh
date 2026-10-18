#!/bin/zsh
# Build FitCastSensor for the connected iPhone, install it and launch it with console output.
# Usage: scripts/ios_deploy.sh [udid]      (default: first iPhone devicectl lists)
set -e
cd "$(dirname "$0")/../ios"
UDID="${1:-$(xcrun devicectl list devices --json-output /tmp/fc_devices.json >/dev/null 2>&1; python3 -c "
import json; d=json.load(open('/tmp/fc_devices.json'))['result']['devices']
ph=[x for x in d if 'iPhone' in x.get('hardwareProperties',{}).get('productType','')]
print(ph[0]['hardwareProperties']['udid'] if ph else '')")}"
[ -n "$UDID" ] || { echo "no iPhone found by devicectl (plug in by USB, unlock, trust)"; exit 2; }
echo "device udid: $UDID"
xcodegen generate >/dev/null
xcodebuild -project RadAR.xcodeproj -scheme FitCastSensor -configuration Debug \
  -destination "id=$UDID" -derivedDataPath build -allowProvisioningUpdates build 2>&1 | grep -E "error|warning: .*(sign|provision)|BUILD" || true
APP=build/Build/Products/Debug-iphoneos/FitCastSensor.app
[ -d "$APP" ] || { echo "build failed: $APP missing"; exit 1; }
xcrun devicectl device install app --device "$UDID" "$APP"
xcrun devicectl device process launch --device "$UDID" --terminate-existing --console com.aadit.fitcast.sensor
