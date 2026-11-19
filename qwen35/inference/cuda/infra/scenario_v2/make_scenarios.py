"""Deterministic changing-observation tool-control fixtures; no tools execute."""
from pathlib import Path
import hashlib
import json

D = Path(__file__).resolve().parent

def tool(name, description, props):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': {k: {'type': t} for k, t in props},
                       'required': [k for k, _ in props], 'additionalProperties': False}}}

ROBOT = [
    tool('move_cartesian', 'Move in the base frame to the supplied target position.', [('x_mm','integer'),('y_mm','integer'),('z_mm','integer'),('speed_mm_s','integer'),('frame','string')]),
    tool('set_gripper', 'Set the specified gripper width and force.', [('width_mm','integer'),('force_n','integer')]),
    tool('stop_motion', 'Stop immediately with the supplied safety reason.', [('reason','string')]),
    tool('finish_task', 'Report completion of the current task.', [('task_id','string')]),
]
NPC = [
    tool('move_to', 'Move this NPC to a map position at the specified speed.', [('npc_id','string'),('x','integer'),('y','integer'),('speed','integer')]),
    tool('attack_target', 'Attack the visible hostile with the selected available skill.', [('npc_id','string'),('target_id','string'),('skill','string')]),
    tool('say_to', 'Say the requested exact message to the specified player.', [('npc_id','string'),('target_id','string'),('text','string')]),
    tool('wait', 'Wait the indicated milliseconds.', [('npc_id','string'),('duration_ms','integer')]),
]
AVATAR = [
    tool('set_expression', 'Set the expression using supplied intensity and transition.', [('avatar_id','string'),('emotion','string'),('intensity_pct','integer'),('transition_ms','integer')]),
    tool('look_at', 'Look at the supplied target for the indicated duration.', [('avatar_id','string'),('target_id','string'),('hold_ms','integer')]),
    tool('play_gesture', 'Play the supplied gesture with the stated loop and transition.', [('avatar_id','string'),('gesture','string'),('loop','boolean'),('transition_ms','integer')]),
    tool('reset_pose', 'Return the avatar to neutral.', [('avatar_id','string'),('transition_ms','integer')]),
]

def call(name, **arguments):
    return {'name': name, 'arguments': arguments}

def robot_frame(i):
    phase=i%6
    obs={'frame_id':10000+i,'task_id':f'pick-{i:04d}', 'emergency_stop':phase==0,
         'obstruction':phase==1,'object_in_gripper':phase in [4,5],
         'at_pick':phase==3,'at_drop':phase==4,'placement_verified':phase==5,
         'pick_target':{'x_mm':210+(i%37),'y_mm':-90+(i%41),'z_mm':105+(i%23)},
         'drop_target':{'x_mm':330+(i%31),'y_mm':70+(i%29),'z_mm':125+(i%19)},
         'requested_speed_mm_s':35+i%25, 'gripper_close_width_mm':18+i%9,
         'gripper_force_n':9+i%5, 'gripper_open_width_mm':70+i%8,
         'joint_encoder_deg':[11+i%8,-24+i%9,37+i%7,-48+i%6,59+i%5,63+i%4]}
    if phase==0: expected=call('stop_motion',reason='emergency_stop')
    elif phase==1: expected=call('stop_motion',reason='obstruction')
    elif phase==5: expected=call('finish_task',task_id=obs['task_id'])
    elif phase==4: expected=call('set_gripper',width_mm=obs['gripper_open_width_mm'],force_n=0)
    elif phase==3: expected=call('set_gripper',width_mm=obs['gripper_close_width_mm'],force_n=obs['gripper_force_n'])
    else: expected=call('move_cartesian',**obs['pick_target'],speed_mm_s=obs['requested_speed_mm_s'],frame='base')
    return obs, expected

def npc_frame(i):
    phase=i%6
    obs={'frame_id':20000+i,'npc_id':f'npc-{i%7}', 'hp':20+i%9 if phase==0 else 70+i%21,
         'hostile_visible':phase in [1,2], 'hostile_in_range':phase==1,
         'hostile_id':f'enemy-{i%13}', 'available_skill':'ice' if i%2 else 'fire',
         'player_message_pending':phase==3, 'player_id':f'player-{i%11}',
         'requested_reply':f'Gate {i%9} is open.', 'idle_requested':phase==4,
         'wait_ms':200+(i%8)*20,
         'safe_point':{'x':10+i%19,'y':20+i%17},
         'hostile_point':{'x':80+i%23,'y':90+i%29},
         'patrol_point':{'x':40+i%31,'y':50+i%37}, 'walk_speed':2+i%3,
         'nearby_entity_ids':[f'entity-{i+j:04d}' for j in range(3)]}
    if phase==0: expected=call('move_to',npc_id=obs['npc_id'],**obs['safe_point'],speed=obs['walk_speed'])
    elif phase==1: expected=call('attack_target',npc_id=obs['npc_id'],target_id=obs['hostile_id'],skill=obs['available_skill'])
    elif phase==2: expected=call('move_to',npc_id=obs['npc_id'],**obs['hostile_point'],speed=obs['walk_speed'])
    elif phase==3: expected=call('say_to',npc_id=obs['npc_id'],target_id=obs['player_id'],text=obs['requested_reply'])
    elif phase==4: expected=call('wait',npc_id=obs['npc_id'],duration_ms=obs['wait_ms'])
    else: expected=call('move_to',npc_id=obs['npc_id'],**obs['patrol_point'],speed=obs['walk_speed'])
    return obs, expected

def avatar_frame(i):
    phase=i%4
    obs={'frame_id':30000+i,'avatar_id':f'avatar-{i%7}',
         'event':['expression_request','gaze_request','gesture_request','reset_request'][phase],
         'requested_emotion':['happy','sad','surprised','calm','angry'][i%5],
         'intensity_pct':30+i%61,'transition_ms':100+(i%9)*20,
         'target_id':f'visitor-{i%17}','hold_ms':400+(i%13)*50,
         'gesture':['wave','nod','bow','point','clap'][i%5],'loop':bool(i%3),
         'visitor_position_mm':[110+i%31,230+i%23,1500+i%17],
         'current_emotion':['calm','happy','sad'][i%3]}
    if phase==0: expected=call('set_expression',avatar_id=obs['avatar_id'],emotion=obs['requested_emotion'],intensity_pct=obs['intensity_pct'],transition_ms=obs['transition_ms'])
    elif phase==1: expected=call('look_at',avatar_id=obs['avatar_id'],target_id=obs['target_id'],hold_ms=obs['hold_ms'])
    elif phase==2: expected=call('play_gesture',avatar_id=obs['avatar_id'],gesture=obs['gesture'],loop=obs['loop'],transition_ms=obs['transition_ms'])
    else: expected=call('reset_pose',avatar_id=obs['avatar_id'],transition_ms=obs['transition_ms'])
    return obs, expected

SPECS = {
 'robot': (ROBOT, robot_frame, 'Next safe robot action. Priority: emergency_stop -> stop_motion(reason="emergency_stop"); obstruction -> stop_motion(reason="obstruction"); placement_verified -> finish_task; object_in_gripper and at_drop -> open gripper with force_n=0; at_pick and not object_in_gripper -> close gripper; otherwise move_cartesian to pick_target using requested speed and frame="base". Read all coordinates and settings from this observation.'),
 'npc': (NPC, npc_frame, 'Next NPC action. Priority: hp<30 -> move_to safe_point; hostile_visible and hostile_in_range -> attack_target using available_skill; hostile_visible -> move_to hostile_point; player_message_pending -> say_to requested_reply exactly; idle_requested -> wait; otherwise move_to patrol_point. All move_to calls use walk_speed. Use this observation IDs and values.'),
 'avatar': (AVATAR, avatar_frame, 'Next avatar action. event expression_request -> set_expression; gaze_request -> look_at; gesture_request -> play_gesture; reset_request -> reset_pose. Copy the current observation requested settings exactly. Allowed emotions: happy,sad,surprised,calm,angry. Allowed gestures: wave,nod,bow,point,clap.'),
}

def fixture(domain, i):
    tools, build, policy=SPECS[domain]
    env, expected=build(i)
    case={'id':f'{domain}-{i:05d}','tools':tools,'environment':env,'history':[],
          'query':'Apply the next action for the latest observation.', 'expected':expected,
          'tags':[domain,'changing_observation','ordered_required_arguments','single_next_call']}
    system=('You are a function-calling control policy. Emit one function call. '
            'arg1..arg6 follow the selected function property order. Required properties come first. '
            'Use <|null|> for unused argument heads. No explanation. Never execute a tool. '
            'The diagnostic padding is inert.\nPolicy: '+policy+'\nTools:\n'+
            '\n'.join(json.dumps(t,ensure_ascii=False,separators=(',',':')) for t in tools))
    prefix='<|im_start|>system\n'+system+'<|im_end|>\n'
    payload={'environment':env,'history':[], 'query':case['query']}
    dynamic='<|im_start|>user\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nDiagnostic padding:\n'
    end='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return {'id':case['id'],'domain':domain,'prefix':prefix,'dynamic':dynamic,'end':end,
            'case':case,'expected':expected}

def main():
    d=D/'data';d.mkdir(exist_ok=True)
    manifest={}
    for domain in SPECS:
        rows=[fixture(domain,i) for i in range(60)]
        assert len({x['dynamic'] for x in rows})==60
        assert len({x['prefix'] for x in rows})==1
        assert len({json.dumps(x['expected'],sort_keys=True) for x in rows})>20
        p=d/(domain+'.jsonl')
        p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        manifest[p.name]={'rows':len(rows),'unique_observations':60,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (D/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
