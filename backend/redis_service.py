import os
import json
import redis
from typing import Dict, Any, Optional

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class RedisService:
    def __init__(self):
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=2
            )
            self.client.ping()
            print(f"[+] Connected to Redis Server at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"[!] Redis connection unavailable ({e}). Operating in memory-fallback mode.")
            self.client = None

    def get_cached_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            cache_key = f"weather:{lat:.2f}:{lon:.2f}"
            cached_val = self.client.get(cache_key)
            if cached_val:
                print(f"[+] Redis Cache HIT for weather key: {cache_key}")
                return json.loads(cached_val)
        except Exception as e:
            print("[!] Redis get_cached_weather error:", e)
        return None

    def set_cached_weather(self, lat: float, lon: float, weather_data: Dict[str, Any], ttl_seconds: int = 300):
        if not self.client:
            return
        try:
            cache_key = f"weather:{lat:.2f}:{lon:.2f}"
            self.client.setex(cache_key, ttl_seconds, json.dumps(weather_data))
            print(f"[+] Weather cached in Redis for {ttl_seconds}s (key: {cache_key})")
        except Exception as e:
            print("[!] Redis set_cached_weather error:", e)

    def publish_emergency_alert(self, alert_payload: Dict[str, Any]):
        """
        Publishes critical rockfall hazard alerts (>60-80% risk) to Redis Pub/Sub channel.
        """
        channel = "rockfall_emergency_alerts"
        if not self.client:
            print(f"[*] (Local Mode) Alert triggered: Mine #{alert_payload.get('mine_id')} Risk: {alert_payload.get('risk_percentage')}%")
            return
        try:
            msg = json.dumps(alert_payload)
            self.client.publish(channel, msg)
            print(f"[+] Published Emergency Alert to Redis channel '{channel}': {msg}")
        except Exception as e:
            print("[!] Redis publish_emergency_alert error:", e)

    def is_connected(self) -> bool:
        if not self.client:
            return False
        try:
            return self.client.ping()
        except Exception:
            return False

redis_service = RedisService()
