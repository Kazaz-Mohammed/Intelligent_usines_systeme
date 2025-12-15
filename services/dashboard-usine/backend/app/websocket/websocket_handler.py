"""
WebSocket endpoint handler
"""
import logging
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# Import will be done at runtime to avoid circular dependency
_websocket_manager = None

def set_websocket_manager(manager):
    """Set websocket manager instance"""
    global _websocket_manager
    _websocket_manager = manager

async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler - broadcast only, no need to receive from client"""
    global _websocket_manager
    
    # Log immediately when handler is called
    logger.info(f">>> WebSocket handler called! Client: {websocket.client}")
    print(f">>> WebSocket handler called! Client: {websocket.client}", flush=True)
    
    if _websocket_manager is None:
        from app.main import websocket_manager
        _websocket_manager = websocket_manager
        logger.info("WebSocket manager was None, imported from app.main")
    
    try:
        await _websocket_manager.connect(websocket)
        logger.info("WebSocket connection accepted successfully")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}", exc_info=True)
        raise
    logger.info("WebSocket client connected, sending welcome message")
    
    # Send welcome message
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to dashboard service",
            "status": "connected"
        })
    except Exception as e:
        logger.error(f"Could not send welcome message: {e}", exc_info=True)
        if _websocket_manager:
            await _websocket_manager.disconnect(websocket)
        return
    
    # Keep connection alive - wait for disconnect or handle optional messages
    # Since this is broadcast-only, we don't need to actively receive
    try:
        import asyncio
        while True:
            try:
                # Use a timeout to periodically check connection status
                # This allows us to keep the connection alive without blocking indefinitely
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                # Check if it's a disconnect message
                if message.get("type") == "websocket.disconnect":
                    logger.debug("Disconnect message received")
                    break
                
                # Handle optional client messages (ping, etc.)
                if "text" in message:
                    data = message["text"]
                    logger.debug(f"Message texte reçu: {data}")
                    try:
                        import json
                        parsed = json.loads(data)
                        logger.debug(f"Message JSON: {parsed}")
                        # Handle ping/pong
                        if parsed.get("type") == "ping":
                            await websocket.send_json({"type": "pong", "message": "Connection alive"})
                    except json.JSONDecodeError:
                        logger.debug(f"Message texte brut: {data}")
                
                elif "bytes" in message:
                    logger.debug("Message binaire reçu")
                    
            except asyncio.TimeoutError:
                # Timeout is normal - connection is still alive, just no message received
                # Send a keepalive ping
                try:
                    await websocket.send_json({"type": "ping", "message": "Keepalive"})
                except Exception:
                    # Connection might be closed
                    break
            except WebSocketDisconnect:
                # Client disconnected normally
                break
            except RuntimeError as e:
                # Handle "Cannot call receive once a disconnect message has been received"
                if "disconnect" in str(e).lower():
                    logger.debug("Disconnect detected via RuntimeError")
                    break
                raise  # Re-raise if it's a different RuntimeError
            except Exception as receive_error:
                logger.debug(f"Error in receive: {receive_error}")
                break
                
    except WebSocketDisconnect:
        pass  # Normal disconnect
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            logger.debug("Disconnect detected")
        else:
            logger.error(f"RuntimeError in WebSocket: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Erreur dans le WebSocket: {e}", exc_info=True)
    finally:
        if _websocket_manager:
            await _websocket_manager.disconnect(websocket)
        logger.info("Client WebSocket déconnecté")

