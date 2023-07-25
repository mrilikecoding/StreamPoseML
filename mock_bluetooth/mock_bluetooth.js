const bleno = require('bleno');

const Descriptor = bleno.Descriptor;
const Characteristic = bleno.Characteristic;
const PrimaryService = bleno.PrimaryService;

const SERVICE_UUID = '0000FFE0-0000-1000-8000-00805F9B34FB';
const CHARACTERISTIC_UUID = '0000FFE1-0000-1000-8000-00805F9B34FB';
const DEVICE_NAME = 'HMSoft';

const EXPECTED_INPUT = 'a';
const SUCCESS_RESPONSE = 'd';

bleno.on('stateChange', function (state) {
    console.log('on -> stateChange: ' + state);

    if (state === 'poweredOn') {
        bleno.startAdvertising(DEVICE_NAME, [SERVICE_UUID]);
    } else {
        bleno.stopAdvertising();
    }
});

bleno.on('advertisingStart', function (error) {
    console.log('on -> advertisingStart: ' + (error ? 'error ' + error : 'success'));

    if (!error) {
        bleno.setServices([
            new PrimaryService({
                uuid: SERVICE_UUID,
                characteristics: [
                    new Characteristic({
                        uuid: CHARACTERISTIC_UUID,
                        properties: ['read', 'write'],
                        onWriteRequest: function (data, offset, withoutResponse, callback) {
                            // handle write requests
                            // for simplicity we'll assume that data is a single byte Buffer
                            if (data.toString() === EXPECTED_INPUT) {
                                console.log(`Received '${EXPECTED_INPUT}', responding with '${SUCCESS_RESPONSE}'`);
                                this.value = Buffer.from(SUCCESS_RESPONSE);
                            }
                            callback(this.RESULT_SUCCESS);
                        },
                        onReadRequest: function (offset, callback) {
                            // handle read requests
                            callback(this.RESULT_SUCCESS, this.value);
                        },
                        descriptors: [
                            new Descriptor({
                                uuid: '2901',
                                value: 'Mock ' + DEVICE_NAME + ' Characteristic'
                            })
                        ]
                    })
                ]
            })
        ]);
    }
});
