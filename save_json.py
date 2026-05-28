# save_json.py

import json
import os

buffer = []

def save_frame_json(json_path, frame_data):
    global buffer

    os.makedirs( os.path.dirname(json_path), exist_ok=True )

    buffer.append( frame_data )
    
    if len(buffer) >= 30:

        with open(json_path, "a", encoding="utf-8") as f:
            for data in buffer:
                f.write(
                    json.dumps(
                        data, 
                        ensure_ascii=False
                    ) +"\n"
                )
        buffer = []

def flush_json(json_path):
    global buffer

    if len(buffer) > 0:
        os.makedirs( os.path.dirname(json_path), exist_ok=True )
        
        with open(json_path, "a", encoding="utf-8") as f:
            for data in buffer:
                f.write( json.dumps( 
                    data, ensure_ascii=False
                 ) + "\n" )
        buffer = []