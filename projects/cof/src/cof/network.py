"""
Network protocol implementation for cof distributed version control.
Implements UDP-based protocol with packet fragmentation and reliability.
"""

import asyncio
import json
import socket
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PacketType(Enum):
    """Network packet types."""
    HANDSHAKE = 0x01
    HANDSHAKE_ACK = 0x02
    AUTH_REQUEST = 0x03
    AUTH_RESPONSE = 0x04
    DATA = 0x05
    DATA_ACK = 0x06
    OBJECT_REQUEST = 0x07
    OBJECT_RESPONSE = 0x08
    REF_REQUEST = 0x09
    REF_RESPONSE = 0x0A
    PUSH_REQUEST = 0x0B
    PUSH_RESPONSE = 0x0C
    ERROR = 0xFF


@dataclass
class NetworkPacket:
    """Network packet structure."""
    packet_type: PacketType
    session_id: str
    sequence: int
    total_packets: int
    payload: bytes
    checksum: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Calculate packet checksum."""
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of packet data."""
        data = self.pack_without_checksum()
        return hashlib.sha256(data).hexdigest()[:16]

    def pack_without_checksum(self) -> bytes:
        """Pack packet without checksum for calculation."""
        header = struct.pack(
            "!B16sII",
            self.packet_type.value,
            self.session_id.encode('utf-8')[:16],
            self.sequence,
            self.total_packets
        )
        return header + self.payload

    def pack(self) -> bytes:
        """Pack complete packet."""
        data = self.pack_without_checksum()
        checksum_bytes = self.checksum.encode('utf-8')
        return checksum_bytes + data

    @classmethod
    def unpack(cls, data: bytes) -> "NetworkPacket":
        """Unpack packet from bytes."""
        if len(data) < 25:  # Minimum packet size
            raise ValueError("Packet too small")

        checksum = data[:16].decode('utf-8')
        packet_data = data[16:]

        # Verify checksum
        calculated_checksum = hashlib.sha256(packet_data).hexdigest()[:16]
        if checksum != calculated_checksum:
            raise ValueError("Packet checksum mismatch")

        if len(packet_data) < 25:
            raise ValueError("Packet data too small")

        packet_type_val, session_id_bytes, sequence, total_packets = struct.unpack(
            "!B16sII", packet_data[:25]
        )
        
        payload = packet_data[25:]
        
        session_id = session_id_bytes.decode('utf-8').rstrip('\x00')
        packet_type = PacketType(packet_type_val)

        packet = cls(
            packet_type=packet_type,
            session_id=session_id,
            sequence=sequence,
            total_packets=total_packets,
            payload=payload
        )
        packet.checksum = checksum
        return packet


@dataclass
class RemoteRepository:
    """Remote repository configuration."""
    name: str
    url: str
    host: str
    port: int
    protocol: str = "udp"
    
    @classmethod
    def from_url(cls, name: str, url: str) -> "RemoteRepository":
        """Create remote from URL."""
        if url.startswith("cof://"):
            url = url[6:]
        elif not url.startswith("udp://"):
            url = f"udp://{url}"
        
        if url.startswith("udp://"):
            url = url[6:]
        
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host = url
                port = 7357  # Default cof port
        else:
            host = url
            port = 7357
        
        return cls(
            name=name,
            url=f"cof://{host}:{port}",
            host=host,
            port=port,
            protocol="udp"
        )


class CofProtocolError(Exception):
    """Protocol-specific errors."""
    pass


class NetworkClient:
    """UDP network client for cof operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.packet_size = config["network"]["packet_size"]
        self.timeout_ms = config["network"]["timeout_ms"]
        self.max_retries = config["network"]["max_retries"]
        self.socket = None
        self.session_id = str(uuid.uuid4())

    async def __aenter__(self):
        """Async context manager entry."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.timeout_ms / 1000.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.socket:
            self.socket.close()
            self.socket = None

    async def handshake(self, remote: RemoteRepository, auth_token: Optional[str] = None) -> bool:
        """Perform handshake with remote server."""
        try:
            # Send handshake
            handshake_data = {"version": "1.0", "client": "cof"}
            if auth_token:
                handshake_data["auth_token"] = auth_token
            
            handshake_packet = NetworkPacket(
                packet_type=PacketType.HANDSHAKE,
                session_id=self.session_id,
                sequence=0,
                total_packets=1,
                payload=json.dumps(handshake_data).encode()
            )

            await self._send_packet(remote, handshake_packet)
            
            # Wait for handshake ACK
            response = await self._receive_packet(remote)
            return response.packet_type == PacketType.HANDSHAKE_ACK

        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            return False

    async def authenticate(self, remote: RemoteRepository, username: str, password: str) -> Optional[str]:
        """Authenticate with remote server and return token."""
        try:
            auth_data = {
                "username": username,
                "password": password,
                "timestamp": int(time.time())
            }
            
            auth_packet = NetworkPacket(
                packet_type=PacketType.AUTH_REQUEST,
                session_id=self.session_id,
                sequence=0,
                total_packets=1,
                payload=json.dumps(auth_data).encode()
            )

            await self._send_packet(remote, auth_packet)
            
            # Wait for auth response
            response = await self._receive_packet(remote)
            
            if response.packet_type == PacketType.AUTH_RESPONSE:
                auth_result = json.loads(response.payload.decode())
                if auth_result.get("success"):
                    return auth_result.get("token")
                else:
                    logger.error(f"Authentication failed: {auth_result.get('error', 'Unknown error')}")
            
            return None

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    async def _send_packet(self, remote: RemoteRepository, packet: NetworkPacket) -> None:
        """Send a single packet."""
        if not self.socket:
            raise CofProtocolError("Socket not initialized")

        data = packet.pack()
        
        # Fragment if necessary
        if len(data) > self.packet_size:
            await self._send_fragmented(remote, packet)
        else:
            self.socket.sendto(data, (remote.host, remote.port))

    async def _send_fragmented(self, remote: RemoteRepository, packet: NetworkPacket) -> None:
        """Send packet in fragments."""
        data = packet.pack()
        fragment_size = self.packet_size - 50  # Reserve space for fragment header
        
        total_fragments = (len(data) + fragment_size - 1) // fragment_size
        
        for i in range(total_fragments):
            start = i * fragment_size
            end = min(start + fragment_size, len(data))
            fragment_data = data[start:end]
            
            # Add fragment header
            fragment_header = struct.pack("!HH", i, total_fragments)
            fragment_packet = fragment_header + fragment_data
            
            if self.socket:
                self.socket.sendto(fragment_packet, (remote.host, remote.port))
            await asyncio.sleep(0.001)  # Small delay between fragments

    async def _receive_packet(self, remote: RemoteRepository, timeout: Optional[float] = None) -> NetworkPacket:
        """Receive a packet with retry logic."""
        timeout = timeout or (self.timeout_ms / 1000.0)
        
        for attempt in range(self.max_retries):
            try:
                if timeout and self.socket:
                    self.socket.settimeout(timeout)
                
                if not self.socket:
                    raise CofProtocolError("Socket not initialized")
                
                data, addr = self.socket.recvfrom(self.packet_size * 2)
                
                # Check if this is a fragmented packet
                if len(data) > 25 and data[:4] == b'\x00\x00':  # Fragment indicator
                    data = await self._receive_fragments(remote, data)
                
                packet = NetworkPacket.unpack(data)
                
                # Verify session ID
                if packet.session_id != self.session_id:
                    logger.warning(f"Received packet with wrong session ID: {packet.session_id}")
                    continue
                
                return packet

            except socket.timeout:
                logger.warning(f"Receive timeout, attempt {attempt + 1}/{self.max_retries}")
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                continue
        
        raise CofProtocolError("Failed to receive packet after retries")

    async def _receive_fragments(self, remote: RemoteRepository, first_fragment: bytes) -> bytes:
        """Receive and reassemble fragmented packets."""
        # Parse fragment header
        fragment_id, total_fragments = struct.unpack("!HH", first_fragment[:4])
        fragment_data = first_fragment[4:]
        
        fragments = {fragment_id: fragment_data}
        
        # Receive remaining fragments
        while len(fragments) < total_fragments:
            try:
                if not self.socket:
                    raise CofProtocolError("Socket not initialized")
                data, addr = self.socket.recvfrom(self.packet_size * 2)
                
                if len(data) >= 4 and data[:2] == b'\x00\x00':
                    frag_id, _ = struct.unpack("!HH", data[:4])
                    fragments[frag_id] = data[4:]
                    
            except socket.timeout:
                break
        
        # Reassemble fragments
        result = b""
        for i in range(total_fragments):
            if i in fragments:
                result += fragments[i]
            else:
                raise CofProtocolError(f"Missing fragment {i}")
        
        return result

    async def request_object(self, remote: RemoteRepository, object_hash: str) -> Optional[bytes]:
        """Request an object from remote repository."""
        try:
            # Send object request
            request_packet = NetworkPacket(
                packet_type=PacketType.OBJECT_REQUEST,
                session_id=self.session_id,
                sequence=0,
                total_packets=1,
                payload=object_hash.encode()
            )

            await self._send_packet(remote, request_packet)
            
            # Receive object response
            response = await self._receive_packet(remote)
            
            if response.packet_type == PacketType.OBJECT_RESPONSE:
                return response.payload
            elif response.packet_type == PacketType.ERROR:
                error_msg = response.payload.decode()
                logger.error(f"Remote error: {error_msg}")
                return None
            else:
                logger.error(f"Unexpected response type: {response.packet_type}")
                return None

        except Exception as e:
            logger.error(f"Object request failed: {e}")
            return None

    async def request_refs(self, remote: RemoteRepository) -> Dict[str, str]:
        """Request all refs from remote repository."""
        try:
            request_packet = NetworkPacket(
                packet_type=PacketType.REF_REQUEST,
                session_id=self.session_id,
                sequence=0,
                total_packets=1,
                payload=b""
            )

            await self._send_packet(remote, request_packet)
            
            response = await self._receive_packet(remote)
            
            if response.packet_type == PacketType.REF_RESPONSE:
                refs_data = json.loads(response.payload.decode())
                return refs_data
            else:
                logger.error(f"Unexpected response type: {response.packet_type}")
                return {}

        except Exception as e:
            logger.error(f"Refs request failed: {e}")
            return {}

    async def push_objects(self, remote: RemoteRepository, objects: Dict[str, bytes]) -> bool:
        """Push objects to remote repository."""
        try:
            # Send push request with object list
            object_list = list(objects.keys())
            push_request = NetworkPacket(
                packet_type=PacketType.PUSH_REQUEST,
                session_id=self.session_id,
                sequence=0,
                total_packets=len(objects) + 1,
                payload=json.dumps(object_list).encode()
            )

            await self._send_packet(remote, push_request)
            
            # Send objects
            for i, (obj_hash, obj_data) in enumerate(objects.items(), 1):
                obj_packet = NetworkPacket(
                    packet_type=PacketType.DATA,
                    session_id=self.session_id,
                    sequence=i,
                    total_packets=len(objects) + 1,
                    payload=obj_data
                )
                await self._send_packet(remote, obj_packet)
            
            # Wait for push response
            response = await self._receive_packet(remote)
            return response.packet_type == PacketType.PUSH_RESPONSE

        except Exception as e:
            logger.error(f"Push failed: {e}")
            return False


class NetworkServer:
    """UDP network server for cof repository."""

    def __init__(self, repository, config: Dict[str, Any]):
        self.repository = repository
        self.config = config
        self.packet_size = config["network"]["packet_size"]
        self.host = "0.0.0.0"
        self.port = 7357
        self.socket = None
        self.running = False

    async def start(self) -> None:
        """Start the network server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.running = True
        
        logger.info(f"Cof server started on {self.host}:{self.port}")
        
        try:
            while self.running:
                try:
                    if not self.socket:
                        raise CofProtocolError("Socket not initialized")
                    data, addr = self.socket.recvfrom(self.packet_size * 2)
                    asyncio.create_task(self._handle_packet(data, addr))
                except Exception as e:
                    logger.error(f"Server error: {e}")
        finally:
            if self.socket:
                self.socket.close()

    async def stop(self) -> None:
        """Stop the network server."""
        self.running = False
        if self.socket:
            self.socket.close()

    async def _handle_packet(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle incoming packet."""
        try:
            packet = NetworkPacket.unpack(data)
            response = await self._process_packet(packet)
            
            if response and self.socket:
                response_data = response.pack()
                self.socket.sendto(response_data, addr)

        except Exception as e:
            logger.error(f"Packet handling error: {e}")
            # Send error response
            error_packet = NetworkPacket(
                packet_type=PacketType.ERROR,
                session_id="",
                sequence=0,
                total_packets=1,
                payload=str(e).encode()
            )
            if self.socket:
                self.socket.sendto(error_packet.pack(), addr)

    async def _process_packet(self, packet: NetworkPacket) -> Optional[NetworkPacket]:
        """Process packet and return response."""
        try:
            if packet.packet_type == PacketType.HANDSHAKE:
                return NetworkPacket(
                    packet_type=PacketType.HANDSHAKE_ACK,
                    session_id=packet.session_id,
                    sequence=0,
                    total_packets=1,
                    payload=json.dumps({"status": "ok"}).encode()
                )

            elif packet.packet_type == PacketType.OBJECT_REQUEST:
                object_hash = packet.payload.decode()
                obj_data = self.repository._load_object(object_hash)
                
                if obj_data:
                    obj_bytes = json.dumps(obj_data).encode()
                    return NetworkPacket(
                        packet_type=PacketType.OBJECT_RESPONSE,
                        session_id=packet.session_id,
                        sequence=0,
                        total_packets=1,
                        payload=obj_bytes
                    )
                else:
                    return NetworkPacket(
                        packet_type=PacketType.ERROR,
                        session_id=packet.session_id,
                        sequence=0,
                        total_packets=1,
                        payload=f"Object {object_hash} not found".encode()
                    )

            elif packet.packet_type == PacketType.REF_REQUEST:
                refs = {}
                refs_dir = self.repository.cof_dir / "refs" / "heads"
                
                if refs_dir.exists():
                    for ref_file in refs_dir.iterdir():
                        with open(ref_file, "r") as f:
                            refs[ref_file.name] = f.read().strip()
                
                return NetworkPacket(
                    packet_type=PacketType.REF_RESPONSE,
                    session_id=packet.session_id,
                    sequence=0,
                    total_packets=1,
                    payload=json.dumps(refs).encode()
                )

            elif packet.packet_type == PacketType.PUSH_REQUEST:
                # Handle push request - would need more complex logic for real implementation
                return NetworkPacket(
                    packet_type=PacketType.PUSH_RESPONSE,
                    session_id=packet.session_id,
                    sequence=0,
                    total_packets=1,
                    payload=json.dumps({"status": "received"}).encode()
                )

            else:
                return NetworkPacket(
                    packet_type=PacketType.ERROR,
                    session_id=packet.session_id,
                    sequence=0,
                    total_packets=1,
                    payload=f"Unknown packet type: {packet.packet_type}".encode()
                )

        except Exception as e:
            logger.error(f"Packet processing error: {e}")
            return NetworkPacket(
                packet_type=PacketType.ERROR,
                session_id=packet.session_id,
                sequence=0,
                total_packets=1,
                payload=str(e).encode()
            )