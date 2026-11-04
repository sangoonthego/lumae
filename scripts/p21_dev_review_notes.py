"""Transcribe actual dev01 image inspection; these are AI observations, not GT."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_outputs/p21_20261103_01'
E=Path('E:/LUMAE/work/d1000/p21_20261103_01/runs/dev01/sources')
# Each boundary bracket is read from actual quarter/half-second frames.
notes={
'029ed1':[(True,True,[20.5,20.75],[21.5,21.75],'Duet appears after the group montage and cuts to a solo dancer. V4 loses the beginning.')],
'04a193':[(False,False,[],[],'Robot/prop closeup, not a child raising colored pens. Reject semantic hallucination.'),(True,True,[24,24.25],[27.25,27.5],'Corridor montage changes clothing/people while approaching; logo follows 27.25. No proposal may include logo or truncate the sequence.')],
'05cbaa':[(True,True,[10.5,10.75],[11.75,12],'Woman raises/displays dark mechanical part; truck crosses next.'),(True,True,[25.5,25.75],[27.75,28],'Flames ignite and continue until cut at 28; first bright flare is before V4 start.')],
'0a721d':[(True,True,[3.75,4],[5,5.25],'Holding-hands skating shot lies between conversation and next seaside shot.')],
'0bcacd':[(True,True,[6.25,6.5],[8.5,8.75],'Blue rays expand from the glowing ring; text/zoom follows. Bracket concerns the burst, not the preceding static ring.')],
'20c18a':[(True,True,[17.25,17.5],[18,18.25],'External yellow-car skid throws dust; interior driver shots flank it.')],
'3fab57':[(True,False,[],[],'Architectural shot montage has multiple corridor and cinema-seat cuts. No unique bounded physical event is established.')],
'416a38':[(True,True,[20.25,20.5],[21.25,21.5],'Hands reach hedge then contact/lean; cut to different green-shirted man on floor.')],
'513380':[(True,False,[],[],'Car rolls/drifts in several adjacent shots; actual full stop is not demonstrable before interior cut. Complete occurrence unverified.')],
'5d2d39':[(False,False,[],[],'Blonde woman holds cigarette amid feather costume, not cocktail glass. Reject unsupported object and action.')],
'714b04':[(True,True,[13.5,14],[17,17.5],'Hands lather and rinse; cleaning-bottle shot begins at 17.5.')],
'7dd3fa':[(True,True,[23,23.25],[25.5,25.75],'Arms wrap into embrace, then separate and turn to sit. Coarse phase includes extra approach and exit.')],
'864c4e':[(True,False,[14.75,15],[16,16.25],'Shelf pickup shown, but another shelf/product-pickup montage appears at 20.5. Generic woman/package query is not proven unique.')],
'9bc46f':[(True,True,[21.5,21.75],[24.75,25],'Cake appears in closeup then group is shown in doorway; older man follows at 25.')],
'bf5acc':[(True,True,[16.25,16.5],[17.5,17.75],'Hug closeup ends at cut to buoy. V4 must cover full hug shot.')],
'd224f2':[(True,True,[27.25,27.5],[28,28.25],'Hand pushes cable into socket, releases by 28.25; projector wide shot starts at 28.5.')],
'db30f6':[(True,True,[51.25,51.5],[52.5,52.75],'Cushion windup then flight and TV impact; reaction shot follows.')],
'ecdb85':[(True,True,[13.5,13.75],[15.25,15.5],'Palms together and forward bow/greeting then hands lower and man leaves. V4 covers only small middle section.')]
}
out=[]
for obs in json.loads((O/'recovery/development_events.json').read_text()):
 sid=obs['source_id'];folder=E/sid
 bundle={'source_id':sid,'actual_complete_overview_inspected':True,'reason':obs.get('abstention_reason','Actual source and dense pages inspected in this session'),'human_verified':False}
 if obs['events']:
  values=next(v for k,v in notes.items() if sid.startswith(k));assert len(values)==len(obs['events'])
  bundle['inspected_review_grids']=[g['path'] for g in json.loads((folder/'review_grid_manifest.json').read_text())]
  bundle['event_reviews']=[{'query':e['query'],'actual_overview_inspected':True,'actual_qc_inspected':True,'semantic_match':m,'unique_event':u,'complete_occurrence':m and u,'onset_bracket_seconds':a,'offset_bracket_seconds':b,'reason':n,'human_verified':False} for e,(m,u,a,b,n) in zip(obs['events'],values)]
 out.append(bundle)
(O/'recovery/development_reviews.json').write_text(json.dumps(out,indent=2)+'\n')
