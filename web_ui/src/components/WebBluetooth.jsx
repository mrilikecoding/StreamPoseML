import React, { useEffect, useState, useRef } from 'react';
import { GrStatusGoodSmall } from "react-icons/gr";
import { FaBluetooth } from "react-icons/fa";
import { FaCopy } from "react-icons/fa";

// Bluetooth - TODO abstract this out so that we can select what actuator to use
const DEVICE_SERVICE_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_SERVICE_UUID.toLowerCase();
const DEVICE_CHARACTERISTIC_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_CHARACTERISTIC_UUID.toLowerCase();

function WebBluetooth({ classifierResult }) {
    const logOutputRef = useRef();

    const [characteristic, setCharacteristic] = useState(null);
    const [bluetoothStatus, setBluetoothStatus] = useState("Connect to Bluetooth");
    const [bluetoothResponse, setBluetoothResponse] = useState(null);
    const [bluetoothSend, setBluetoothSend] = useState(null);
    const [deviceServiceUUID, setDeviceServiceUUID] = useState(DEVICE_SERVICE_UUID);
    const [deviceCharacteristicUUID, setDeviceCharacteristicUUID] = useState(DEVICE_CHARACTERISTIC_UUID);
    const [sendPositiveString, setSendPositiveString] = useState('a'); 
    const [sendNegativeString, setSendNegativeString] = useState('');
    const [logOutput, setLogOutput] = useState('Nothing to see here... first, connect to a Bluetooth device.', () => {
        console.log('updated');

    });
    const [currentClassification, setCurrentClassification] = useState(null);
    const [bluetoothIndicator, setBluetoothIndicator] = useState(false);


    const handleLogOutputUpdate = (entry) => {
        const updatedOutput = `${logOutput} \n ${new Date().toISOString()} -- ${entry}`; 

        logOutputRef.current.scrollTop = logOutputRef.current.scrollHeight;

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
                handleLogOutputUpdate(bluetoothStatus)
                return device.gatt.connect();
            })
            .then(server => {
                setBluetoothStatus("Discovering service...");
                handleLogOutputUpdate(bluetoothStatus)
                return server.getPrimaryService(deviceServiceUUID);
            })
            .then(service => {
                setBluetoothStatus("Discovering characteristic...");
                handleLogOutputUpdate(bluetoothStatus)
                return service.getCharacteristic(deviceCharacteristicUUID);
            })
            .then(characteristic => {
                setBluetoothStatus("Bluetooth Device connected!");
                handleLogOutputUpdate(bluetoothStatus)
                setCharacteristic(characteristic);
                setBluetoothIndicator(true);
            })
            .catch(error => {
                console.log(error);
                setBluetoothStatus("Failed to connect: " + error.message);
                handleLogOutputUpdate(bluetoothStatus)
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
            setBluetoothSend(`value: ${pos}, encoded: ${value}`);
            if (characteristic) {
                characteristic.writeValueWithoutResponse(value)
                            .then(() => {
                                const log = 'Write operation is complete.';
                                handleLogOutputUpdate(log);
                                return characteristic.readValue();
                            })
                            .then(value => {
                                let decoder = new TextDecoder('utf-8');
                                let result = decoder.decode(value);
                                const log = 'Read operation result: ' +  result;
                                handleLogOutputUpdate(log);
                                setBluetoothResponse(result);
                                setBluetoothStatus("Bluetooth Device connected!");
                                setBluetoothIndicator(true);
                            })
                            .catch(error => {
                                handleLogOutputUpdate(error);
                                setBluetoothResponse("");
                                setBluetoothStatus("Error: Reconnect")
                                setBluetoothIndicator(false);
                            });
            }
        } else {
            let value = encoder.encode(neg);
            setBluetoothSend(`value: ${neg}, encoded: ${value}`);

        }

    }, [classifierResult, characteristic])

    return (
        <div className='prose'>
            <div>
                <div className="card border">
                    <div className="card-body">
                        <div role="alert" className='"alert alert-warning text-warning-content'>
                            {
                                !navigator.bluetooth ? <b>Bluetooth is not supported in this browser. Please try Chrome.</b> : 
                                <>
                                    <button className={"btn w-full mb-4 " + (bluetoothIndicator ? "btn-success" : "btn-warning")} onClick={connectToDevice}>{bluetoothStatus}<FaBluetooth />
                                    </button>
                                </>
                            }
                        </div>
                        <div className='card-title'>
                            Result <GrStatusGoodSmall color={currentClassification ? 'green' : 'red'} /> 
                            Connection <GrStatusGoodSmall color={bluetoothIndicator ? 'green' : 'gray'} /> 
                        </div>
                        Send {bluetoothSend} | Received {bluetoothResponse}
                        <div tabIndex={0} className="collapse collapse-arrow border border-base-300">
                            <input type="checkbox" /> 
                            <div className="collapse-title text-xl font-small">
                               Settings 
                            </div>
                            <div className="collapse-content"> 
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
                            </div>
                        </div>
                        <div tabIndex={0} className="collapse collapse-arrow border border-base-300">
                            <input type="checkbox" /> 
                            <div className="collapse-title text-xl font-small">
                               Logs 
                            </div>
                            <div className="collapse-content"> 
                                {/* <pre className='text-xs overflow-hidden w-3/4 h-36 flex flex-col-reverse overflow-y-scroll overflow-x-scroll'>{logOutput}</pre> */}
                                <pre ref={logOutputRef} className='text-xs w-3/4 overflow-hidden h-36 overflow-y-scroll overflow-x-scroll'>{logOutput}</pre>
                                <div className="join">
                                    <button className="join-item btn btn-sm" onClick={ () => setLogOutput('') }>Clear logs</button>
                                    <button className="join-item btn btn-sm" onClick={ () => navigator.clipboard.writeText(logOutput) }>Copy<FaCopy /></button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )

}

export default WebBluetooth;
