import React, { useEffect, useState } from 'react';

// Bluetooth - TODO abstract this out so that we can select what actuator to use
const DEVICE_SERVICE_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_SERVICE_UUID.toLowerCase();
const DEVICE_CHARACTERISTIC_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_CHARACTERISTIC_UUID.toLowerCase();

function WebBluetooth({ classifierResult }) {
    const [characteristic, setCharacteristic] = useState(null);
    const [bluetoothStatus, setBluetoothStatus] = useState("Connect to Bluetooth");
    const [bluetoothResponse, setBluetoothResponse] = useState(null);
    const [bluetoothSend, setBluetoothSend] = useState(null);
    const [deviceServiceUUID, setDeviceServiceUUID] = useState(DEVICE_SERVICE_UUID);
    const [deviceCharacteristicUUID, setDeviceCharacteristicUUID] = useState(DEVICE_CHARACTERISTIC_UUID);
    const [sendPositiveString, setSendPositiveString] = useState('a'); 
    const [sendNegativeString, setSendNegativeString] = useState('');
    const [logOutput, setLogOutput] = useState('');
    const [currentClassification, setCurrentClassification] = useState(null);

    const handleLogOutputUpdate = (entry) => {
        const updatedOutput = logOutput + "/n" + Date.now().toString() + entry; 
        setLogOutput(updatedOutput);
    }

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
        if (!classifierResult || classifierResult === null) {
            setCurrentClassification(false);
        } else if (classifierResult.classification === true) {
            setCurrentClassification(true);
        } else {
            setCurrentClassification(false);
        }
    }, [classifierResult]);

    useEffect(() => {
        const encoder = new TextEncoder('utf-8');
        // TODO create way to set these conditions in UI
        const pos = sendPositiveString;
        const neg = sendNegativeString;

        if (classifierResult && classifierResult.classification === true) {
            let value = encoder.encode(pos);
            setBluetoothSend(`${pos} ${value}`);
            if (characteristic) {
                characteristic.writeValue(value)
                            .then(() => {
                                const log = 'Write operation is complete.';
                                handleLogOutputUpdate(log);
                                setLogOutput(logOutput + '\n' + log)
                                // Now read from the characteristic
                                return characteristic.readValue();
                            })
                            .then(value => {
                                let decoder = new TextDecoder('utf-8');
                                let result = decoder.decode(value);
                                const log = 'Read operation result: ' +  result;
                                handleLogOutputUpdate(log);
                                setBluetoothResponse(result);
                            })
                            .catch(error => {
                                handleLogOutputUpdate(error);
                                setBluetoothResponse("");
                            });
            }
        } else {
            let value = encoder.encode(neg);
            setBluetoothSend(`${neg}: ${value}`);

        }

    }, [classifierResult, characteristic])

    return (
        <div className='prose mb-8'>
            <div role="alert" className='"alert alert-warning text-warning-content'>
                {
                    !navigator.bluetooth ? <b>Bluetooth is not supported in this browser. Please try Chrome.</b> : <></>
                }
            </div>
            <button className="btn btn-primary w-full mb-4" onClick={connectToDevice}>{bluetoothStatus}</button>
            <div className="indicator">
                <div className="indicator-item indicator-bottom">
                </div> 
                <div className={"card border " + (currentClassification ? 'ring-4 ring-green-500' : 'ring-4 ring-red-500')}>
                    <div className="card-body">
                        <span className="card-title">Bluetooth Actuation</span> 
                        <label className="text-sm input input-bordered flex items-center gap-2">
                            Service UUID
                            <input 
                                type="text"
                                onInput={ (e) => setDeviceServiceUUID(e.target.value)} 
                                value={deviceServiceUUID} 
                                className="text-sm input w-full" />
                        </label>
                        <label className="text-sm input input-bordered flex items-center gap-2">
                            Characteristic UUIDs 
                            <input 
                                type="text" 
                                onInput={ (e) => setDeviceCharacteristicUUID(e.target.value)}
                                value={deviceCharacteristicUUID} 
                                className="text-sm input w-full" />
                        </label>
                        <label className="text-sm input input-bordered flex items-center gap-2">
                            Send String on Positive Result
                            <input 
                                type="text"
                                onInput={ (e) => setSendPositiveString(e.target.value)} 
                                value={sendPositiveString} 
                                className="text-sm input w-full" />
                        </label>
                        <label className="text-sm input input-bordered flex items-center gap-2">
                            Send String on Negative Result 
                            <input 
                                type="text" 
                                onInput={ (e) => setSendNegativeString(e.target.value)}
                                value={sendNegativeString} 
                                className="text-sm input w-full" />
                        </label>
                        <div className="stat">
                            <div className="stats shadow">
                                <div className="stat">
                                    <div className="stat-figure text-secondary"></div>
                                    <div className="stat-title">Send</div>
                                    <div className="stat-value">{bluetoothSend}</div>
                                </div>
                                <div className="stat">
                                    <div className="stat-figure text-secondary"></div>
                                    <div className="stat-title">Received</div>
                                    <div className="stat-value">{bluetoothResponse}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <textarea className="object-fill textarea textarea-bordered w-full" value={logOutput}></textarea>
                </div>
            </div>
        </div>
    )

}

export default WebBluetooth;
