"""Basic connection example.
"""
import os

import redis
from dotenv import load_dotenv

load_dotenv()

r = redis.Redis(
    host='redis-17655.c1.asia-northeast1-1.gce.redns.redis-cloud.com',
    port=17655,
    decode_responses=True,
    username="default",
    password=os.getenv("REDIS_PASSWORD")
)

success = r.set('foo', 'bar')
# True

result = r.get('foo')
print(result)
# >>> bar

