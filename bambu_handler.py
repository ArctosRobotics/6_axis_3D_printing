"""
Bambu Lab 3D Printer Handler
MQTT-based communication with Bambu Lab printers

This module provides real-time monitoring and control of Bambu Lab 3D printers
via MQTT protocol. It handles connection management, status monitoring, and
event notifications for automated workflows.

Author: Arctos Studio
Date: December 2024
"""

import json
import ssl
import threading
import time
from typing import Optional, Dict, Any, Callable
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import paho.mqtt.client as mqtt


class BambuPrinterStatus:
    """Enum-like class for printer states"""
    UNKNOWN = "unknown"
    IDLE = "idle"
    PRINTING = "printing"
    PAUSED = "paused"
    FINISH = "finish"
    FAILED = "failed"
    OFFLINE = "offline"


class BambuHandler(QObject):
    """
    Handler for Bambu Lab 3D printer MQTT communication
    
    Connects to Bambu Lab printers via MQTT to monitor print status,
    progress, and trigger automated workflows when prints complete.
    
    Signals:
        connection_changed(bool): Emitted when connection state changes
        status_changed(str): Emitted when printer status changes
        print_progress(int): Emitted with print progress (0-100)
        print_complete(): Emitted when print finishes successfully
        print_failed(str): Emitted when print fails with error message
        error_occurred(str): Emitted on communication errors
        message_received(dict): Emitted with raw MQTT message data
    """
    
    # Signals
    connection_changed = pyqtSignal(bool)  # Connected/Disconnected
    status_changed = pyqtSignal(str)  # Status string
    print_progress = pyqtSignal(int)  # Progress 0-100
    print_complete = pyqtSignal()  # Emitted when print finishes
    print_failed = pyqtSignal(str)  # Emitted on failure with reason
    error_occurred = pyqtSignal(str)  # Error message
    message_received = pyqtSignal(dict)  # Raw message data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Connection parameters
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.ip_address = ""
        self.serial_number = ""
        self.access_code = ""
        
        # Status tracking
        self.current_status = BambuPrinterStatus.UNKNOWN
        self.print_percentage = 0
        self.current_file = ""
        self.layer_num = 0
        self.total_layers = 0
        self.remaining_time = 0  # seconds
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        self._connection_lock = threading.Lock()
        
        # Callbacks for blocking wait
        self._wait_callbacks: list = []
        
        # Reconnection
        self._reconnect_timer = QTimer()
        self._reconnect_timer.timeout.connect(self._attempt_reconnect)
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        
        print("[BambuHandler] Initialized")
    
    def connect(self, ip: str, serial: str, access_code: str) -> bool:
        """
        Connect to Bambu Lab printer via MQTT
        
        Args:
            ip: Printer IP address (e.g., "192.168.1.100")
            serial: Printer serial number (e.g., "AC12309BH109")
            access_code: Access code from printer settings
            
        Returns:
            bool: True if connection initiated successfully
        """
        with self._connection_lock:
            # Disconnect if already connected
            if self.connected:
                self.disconnect()
            
            # Store connection parameters
            self.ip_address = ip
            self.serial_number = serial
            self.access_code = access_code
            
            try:
                # Create MQTT client
                self.client = mqtt.Client(client_id=f"arctos_studio_{int(time.time())}")
                
                # Set username and password
                self.client.username_pw_set("bblp", access_code)
                
                # Set callbacks
                self.client.on_connect = self._on_connect
                self.client.on_disconnect = self._on_disconnect
                self.client.on_message = self._on_message
                
                # Enable TLS (Bambu Lab uses encrypted MQTT)
                self.client.tls_set(cert_reqs=ssl.CERT_NONE)
                self.client.tls_insecure_set(True)
                
                # Connect to printer
                print(f"[BambuHandler] Connecting to {ip}:8883...")
                self.client.connect(ip, 8883, 60)
                
                # Start network loop in background thread
                self.client.loop_start()
                
                return True
                
            except Exception as e:
                error_msg = f"Failed to connect to Bambu Lab printer: {str(e)}"
                print(f"[BambuHandler ERROR] {error_msg}")
                self.error_occurred.emit(error_msg)
                return False
    
    def disconnect(self):
        """Disconnect from printer"""
        with self._connection_lock:
            self._stop_monitoring = True
            
            if self.client:
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                    print("[BambuHandler] Disconnected from printer")
                except Exception as e:
                    print(f"[BambuHandler ERROR] Disconnect error: {e}")
                finally:
                    self.client = None
            
            self.connected = False
            self.connection_changed.emit(False)
            self._reconnect_timer.stop()
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print("[BambuHandler] Connected successfully")
            self.connected = True
            self.connection_changed.emit(True)
            self._reconnect_attempts = 0
            
            # Subscribe to printer status topic
            topic = f"device/{self.serial_number}/report"
            client.subscribe(topic)
            print(f"[BambuHandler] Subscribed to {topic}")
            
            # Request initial status
            self._request_status()
            
        else:
            error_msg = f"Connection failed with code {rc}"
            print(f"[BambuHandler ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            self.connected = False
            self.connection_changed.emit(False)
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        print(f"[BambuHandler] Disconnected (code: {rc})")
        self.connected = False
        self.connection_changed.emit(False)
        
        # Attempt reconnection if unexpected disconnect
        if rc != 0 and self._reconnect_attempts < self._max_reconnect_attempts:
            print(f"[BambuHandler] Scheduling reconnection attempt {self._reconnect_attempts + 1}")
            self._reconnect_timer.start(5000)  # Try reconnect after 5 seconds
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to printer"""
        self._reconnect_timer.stop()
        self._reconnect_attempts += 1
        
        print(f"[BambuHandler] Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
        
        if self.ip_address and self.serial_number and self.access_code:
            self.connect(self.ip_address, self.serial_number, self.access_code)
        else:
            print("[BambuHandler ERROR] Cannot reconnect: missing connection parameters")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received from printer"""
        try:
            # Parse JSON message
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Emit raw message
            self.message_received.emit(payload)
            
            # Extract print information
            print_data = payload.get('print', {})
            
            # Update status
            old_status = self.current_status
            gcode_state = print_data.get('gcode_state', '')
            
            # Map gcode_state to our status enum
            if gcode_state == 'IDLE':
                self.current_status = BambuPrinterStatus.IDLE
            elif gcode_state == 'RUNNING':
                self.current_status = BambuPrinterStatus.PRINTING
            elif gcode_state == 'PAUSE':
                self.current_status = BambuPrinterStatus.PAUSED
            elif gcode_state == 'FINISH':
                self.current_status = BambuPrinterStatus.FINISH
            elif gcode_state == 'FAILED':
                self.current_status = BambuPrinterStatus.FAILED
            else:
                self.current_status = BambuPrinterStatus.UNKNOWN
            
            # Emit status change if changed
            if old_status != self.current_status:
                print(f"[BambuHandler] Status changed: {old_status} -> {self.current_status}")
                self.status_changed.emit(self.current_status)
                
                # Emit specific signals
                if self.current_status == BambuPrinterStatus.FINISH:
                    print("[BambuHandler] Print completed!")
                    self.print_complete.emit()
                    self._trigger_wait_callbacks(True)
                    
                elif self.current_status == BambuPrinterStatus.FAILED:
                    error_msg = print_data.get('fail_reason', 'Unknown error')
                    print(f"[BambuHandler] Print failed: {error_msg}")
                    self.print_failed.emit(error_msg)
                    self._trigger_wait_callbacks(False)
            
            # Update progress
            old_progress = self.print_percentage
            self.print_percentage = print_data.get('mc_percent', 0)
            
            if old_progress != self.print_percentage:
                self.print_progress.emit(self.print_percentage)
            
            # Update other info
            self.current_file = print_data.get('subtask_name', '')
            self.layer_num = print_data.get('layer_num', 0)
            self.total_layers = print_data.get('total_layer_num', 0)
            self.remaining_time = print_data.get('mc_remaining_time', 0)
            
        except json.JSONDecodeError as e:
            print(f"[BambuHandler ERROR] Failed to parse message: {e}")
        except Exception as e:
            print(f"[BambuHandler ERROR] Message processing error: {e}")
    
    def _request_status(self):
        """Request current status from printer"""
        if not self.connected or not self.client:
            return
        
        try:
            # Publish status request
            topic = f"device/{self.serial_number}/request"
            payload = json.dumps({"pushing": {"sequence_id": "0", "command": "pushall"}})
            self.client.publish(topic, payload)
            print("[BambuHandler] Requested status update")
        except Exception as e:
            print(f"[BambuHandler ERROR] Failed to request status: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current printer status
        
        Returns:
            dict: Status information including:
                - connected: bool
                - status: str (idle, printing, etc.)
                - progress: int (0-100)
                - file: str (current file name)
                - layer: str (current/total layers)
                - remaining_time: int (seconds)
        """
        return {
            'connected': self.connected,
            'status': self.current_status,
            'progress': self.print_percentage,
            'file': self.current_file,
            'layer': f"{self.layer_num}/{self.total_layers}",
            'remaining_time': self.remaining_time,
            'ip': self.ip_address
        }
    
    def wait_for_print_complete(self, timeout: int = 3600) -> bool:
        """
        Block until print completes or timeout
        
        Args:
            timeout: Maximum wait time in seconds (default: 1 hour)
            
        Returns:
            bool: True if print completed successfully, False on timeout/failure
        """
        if not self.connected:
            print("[BambuHandler ERROR] Not connected to printer")
            return False
        
        # If already finished, return immediately
        if self.current_status == BambuPrinterStatus.FINISH:
            return True
        
        # If failed, return immediately
        if self.current_status == BambuPrinterStatus.FAILED:
            return False
        
        # Set up wait callback
        result = {'completed': False, 'success': False}
        event = threading.Event()
        
        def callback(success):
            result['completed'] = True
            result['success'] = success
            event.set()
        
        self._wait_callbacks.append(callback)
        
        # Wait for event or timeout
        print(f"[BambuHandler] Waiting for print completion (timeout: {timeout}s)...")
        event.wait(timeout)
        
        # Remove callback
        if callback in self._wait_callbacks:
            self._wait_callbacks.remove(callback)
        
        if result['completed']:
            print(f"[BambuHandler] Wait completed: {'SUCCESS' if result['success'] else 'FAILED'}")
            return result['success']
        else:
            print("[BambuHandler] Wait timed out")
            return False
    
    def _trigger_wait_callbacks(self, success: bool):
        """Trigger all registered wait callbacks"""
        for callback in self._wait_callbacks[:]:  # Copy list to avoid modification during iteration
            try:
                callback(success)
            except Exception as e:
                print(f"[BambuHandler ERROR] Callback error: {e}")
    
    def is_printing(self) -> bool:
        """Check if printer is currently printing"""
        return self.current_status == BambuPrinterStatus.PRINTING
    
    def is_idle(self) -> bool:
        """Check if printer is idle"""
        return self.current_status == BambuPrinterStatus.IDLE
    
    def get_progress(self) -> int:
        """Get current print progress (0-100)"""
        return self.print_percentage
    
    def get_remaining_time(self) -> int:
        """Get estimated remaining time in seconds"""
        return self.remaining_time
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()
