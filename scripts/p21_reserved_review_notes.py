"""Actual reserved01 dense-frame observations, fixed annotation protocol."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_outputs/p21_20261103_01';E=Path('E:/LUMAE/work/d1000/p21_20261103_01/runs/reserved01/sources')
# semantic, unique+complete supported, onset bracket, offset bracket, actual reason
NOTES={
4:(True,False,[10,10.25],[11.25,11.5],'Plane overhead is followed by further plane-flight shots at 13.25 and later. Generic plane-flight query is not proven a unique full occurrence.'),
6:(True,True,[19,19.25],[19.75,20],'Hands lift at19.25, mask removed by20. V4 starts before removal and other proposals extend to text/portrait context.'),
9:(True,False,[],[],'Cabinet opens in a dark wide view then cut reveals boy already holding/eating cookies. Actual taking-out motion is not clearly visible; do not label inferred action.'),
10:(True,True,[16,16.5],[19,19.5],'Two tiles first appear16.5, four18, six19.5. Expansion completes before long static six-tile display.'),
11:(True,False,[],[],'Wings first spread4.5 and remain animated through close-up beyond7.25, outside bounded proposal context. Full flap episode not established; do not accept partial wings shot.'),
13:(True,True,[12,12.5],[17.5,18],'Competition-stage dancing begins after outdoor rehearsal, ends before car interior. V4 truncates the group dance.'),
16:(True,False,[],[],'Typing closeup begins26 and continues to video end; current evidence does not bracket a completed distinct keystroke episode. V4 retains only late middle.'),
17:(True,True,[34,34.25],[35.25,35.5],'Lipstick application closeup is between portrait and next couple. V4 loses most action.'),
19:(False,False,[],[],'No beer pouring from bottle is shown in dense region. Tap/tilted glass20.5 then filled glass placed23; generated pour assertion is unsupported.'),
20:(True,True,[16.5,16.75],[18.25,18.5],'Washing hands shot occurs after masked face and before gardening. Coarse phase missed the late end; raw v4 retained a fragment.'),
21:(True,True,[22.5,23],[26.5,27],'Ambulance recedes continuously in unique street shot; foundation card begins27. Frozen sparse baseline spans full shot.'),
24:(True,True,[22.75,23],[24,24.25],'Divers point downward in second underwater shot, then cut to swimming with a crate. V4 starts during earlier horizontal swim.'),
28:(True,True,[22.75,23],[24,24.25],'Glasses approach and meet during closeup toast; product shop follows24.25. V4 encloses this shot with quarter-second boundary evidence.'),
31:(True,True,[21.25,21.5],[23,23.25],'Dragon opens glowing mouth until cut to black text. V4 ends while mouth opening is ongoing.'),
32:(True,True,[15,15.25],[16.25,16.5],'Hand dips chip then lifts topping; fork shot begins16.5. V4 stops before the chip has completed its lift.'),
35:(True,False,[],[],'Woman lifts camper at22.5 then lowers/holds it and lifts again toward man24.5. Generic lift query does not separate two motions reliably.'),
36:(True,False,[],[],'Bear-shaped ice remains within ice-cube montage; no completed unique melt episode is demonstrated. Current peak29.5 precedes the ice-cube view.'),
37:(True,True,[15.5,16],[23.5,24],'Metal pieces converge into butterfly through montage; shape is complete at24. V4 isolates late surface closeup instead of assembly.'),
38:(True,True,[37.5,37.75],[38.25,38.5],'Flash begins streak37.75, peaks38, fades through38.25. V4 covers only peak and truncates the fading flash.'),
40:(False,False,[],[],'White-coated man uses chopsticks, not a fork. Do not accept object substitution from coarse overview.'),
42:(True,True,[25.75,26],[26.25,26.5],'Skater emerges from tunnel into dust/daylight at26; cut to sunset group26.5. V4 localizes earlier tunnel face.'),
45:(True,True,[13.25,13.5],[15,15.25],'Pit member uses wheel tool; cockpit replaces wheel at15.25. V4 retains only middle half second.'),
46:(True,True,[22.25,22.5],[22.5,22.75],'Hood visible open22 and closed22.5; mechanic celebrates thereafter. V4 peak is earlier overhead car view.'),
48:(False,False,[],[],'Hand visibly grasps and pulls paper from printer130.75-131.75; automatic receipt-feeding action is not established. Do not infer mechanism.'),
50:(True,True,[22,22.5],[24,24.5],'Finger scroll changes website content from22.5 onward; cut to walking group24.5. V4 begins before observed scroll.'),
52:(True,True,[5.5,5.75],[7.5,7.75],'Boat moves right across sunset river; cut to group meal7.75. V4 ends before last boat frame7.5.'),
58:(True,True,[15,15.25],[17.5,17.75],'Knitted pair holds hands and dances through purple-lit shot, followed by train. Sparse frozen baseline covers the visible dance.'),
59:(True,True,[25.25,25.5],[26.5,26.75],'Wax dip and lift occurs25.5-26.5; next view pulls wax string26.75. V4 starts near the end and includes another shot.'),
60:(True,True,[47,47.5],[48,48.5],'Wheel-backed riverside running view only47.5-48.0; earlier street views and later shoe closeup do not satisfy distinctive background. V4 retrieves later feet.'),
61:(True,True,[21.5,21.75],[22.25,22.5],'Helmet-strap adjustment under chin occurs before bridge cut22.5. V4 instead retrieves woman standing/talking later23.125.')
}
def main():
 out=[]
 for obs in json.loads((O/'recovery/reserved_events.json').read_text()):
  sid=obs['source_id'];n=int(obs['recovery_id'].split('_')[1]);folder=E/sid
  bundle={'source_id':sid,'actual_complete_overview_inspected':True,'reason':obs.get('abstention_reason','Actual overview and dense frames inspected'),'human_verified':False}
  if obs['events']:
   m,u,a,b,reason=NOTES[n]
   bundle['inspected_review_grids']=[g['path'] for g in json.loads((folder/'review_grid_manifest.json').read_text())]
   bundle['event_reviews']=[{'query':obs['events'][0]['query'],'actual_overview_inspected':True,'actual_qc_inspected':True,'semantic_match':m,'unique_event':u,'complete_occurrence':m and u,'onset_bracket_seconds':a,'offset_bracket_seconds':b,'reason':reason,'human_verified':False}]
  out.append(bundle)
 assert len(out)==41
 (O/'recovery/reserved_reviews.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
