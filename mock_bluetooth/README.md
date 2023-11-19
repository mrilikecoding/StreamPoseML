# Mock Bluetooth Service

This is for testing StreamPoseML's bluetooth integration. The library `bleno` seems to be a bit outdated so you'll need to run an older version of node, and may even need to install python 2.7 and run within a python 2.7 environment.

This is only tested on Mac OS X. To run, install node version 12. For example, with nvm, you can `nvm install 12` and `nvm use 12`.

You will also need to install XCode.

Then from the root directory here run `npm start`.

The bluetooth connection details are hardcoded directly into the file for now.

## Alternatives

Admittedly it is difficult to test test a web app with bluetooth. One thing to note is that BLE (Bluetooth Low Energy) is way easier to work with in this context. If running this microservice doesn't work for your implementation, I would suggest working with mobile Bluetooth mocking clients, such as LightBlue. 