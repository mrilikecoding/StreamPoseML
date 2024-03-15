import React, { useEffect, useState } from 'react';

function WebBluetooth({deviceServiceUUID, deviceCharacteristicUUID, classifierResult}) {
    const [characteristic, setCharacteristic] = useState(null);
    const [bluetoothStatus, setBluetoothStatus] = useState("Connect to Bluetooth");
    const [bluetoothResponse, setBluetoothResponse] = useState(null);
    const [bluetoothSend, setBluetoothSend] = useState(null);
    
    const connectToDevice = () => {
        setBluetoothStatus("Searching for device...");
        navigator.bluetooth.requestDevice({
            filters: [{ services: [deviceServiceUUID] }],
            optionalServices: [deviceServiceUUID]
        })
            .then(device => {
                setBluetoothStatus("Connecting to device...");
                return device.gatt.connect();
            })
            .then(server => {
                setBluetoothStatus("Discovering service...");
                return server.getPrimaryService(deviceServiceUUID);
            })
            .then(service => {
                setBluetoothStatus("Discovering characteristic...");
                return service.getCharacteristic(deviceCharacteristicUUID);
            })
            .then(characteristic => {
                setBluetoothStatus("Device connected!");
                setCharacteristic(characteristic);
            })
            .catch(error => {
                console.log(error);
                setBluetoothStatus("Failed to connect: " + error.message);
            });
    };

    useEffect(() => {
        const encoder = new TextEncoder('utf-8');
        // TODO create way to set these conditions in UI
        const pos = 'a';
        const neg = '';

        if (classifierResult && classifierResult.classification === true) {
            let value = encoder.encode(pos);
            setBluetoothSend(`${pos} ${value}`);
            if (characteristic) {
                characteristic.writeValue(value)
                            .then(() => {
                                console.log('Write operation is complete.');
                                // Now read from the characteristic
                                return characteristic.readValue();
                            })
                            .then(value => {
                                let decoder = new TextDecoder('utf-8');
                                let result = decoder.decode(value);
                                console.log('Read operation result:', result);
                                setBluetoothResponse(result);
                            })
                            .catch(error => {
                                console.log(error);
                                setBluetoothResponse("");
                            });
            }
        } else {
            let value = encoder.encode(neg);
            setBluetoothSend(`${neg}: ${value}`);

        }

    }, [classifierResult, characteristic])

    return (
        <div>
            <div className='container'>

            <div className='column bg'
                style={{
                    backgroundColor: classifierResult.classification === true ? 'green' : 'red'
                }}
                >
                    <div>
                        {
                            !navigator.bluetooth ? <b>Bluetooth not supported in this browser. Please try Chrome.</b> :
                            <button onClick={connectToDevice}>{bluetoothStatus}</button>
                        }
                    </div>
                    <p>Bluetooth Send: {bluetoothSend}</p>
                    <p>Bluetooth Response: {bluetoothResponse}</p>
                    <p>(see console for read/write stream)</p>
                    <p>Service UUID: {deviceServiceUUID}</p>
                    <p>Characteristic UUID: {deviceCharacteristicUUID}</p>
            </div>
            </div>
        </div>
    )

}

export default WebBluetooth;
