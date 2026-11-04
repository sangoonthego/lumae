"""Actual full-source overview observations after experimental code freeze."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_outputs/p21_20261103_01'
# One bounded new visual event attempt per reserved source, no paraphrase retries.
NOTES={
4:('An airplane flies overhead above a forest.',9,11.5,'airplane','flies overhead','forest'),
6:('A red-haired woman removes a face mask.',18.5,20.5,'woman','removes','face mask'),
9:('A boy reaches into a cabinet and takes out a packet of cookies.',30,33.5,'boy','takes out','packet of cookies'),
10:('A video-call display expands from two participant tiles to six tiles.',15.5,23,'video-call display','expands','participant tiles'),
11:('An animated robin spreads and flaps its wings beside a bottle.',3,5.5,'robin','flaps','wings'),
13:('Children dance together on a brightly lit competition stage.',12,17.5,'children','dance','competition stage'),
14:'Stone-texture montage; no distinct bounded action supported by full-source overview.',
16:('Hands type on a keyboard beneath a blue sleeve.',25,29,'hands','type','keyboard'),
17:('A lipstick applicator moves across a person\'s lower lip.',33,35.5,'applicator','moves across','lower lip'),
19:('Beer pours from a bottle into a tall glass.',22.5,24.5,'beer','pours','glass'),
20:('A person rubs wet hands together under running water.',15,17.5,'person','rubs','wet hands'),
21:('An ambulance drives away down a dark street.',22,27,'ambulance','drives away','street'),
22:'Stationary shopping bags and typography; no physical action justified.',
24:('Scuba divers swim downward above a coral reef.',21.5,25.5,'scuba divers','swim downward','coral reef'),
27:'Repeated closed garage-door photographs; no opening/closing action observed.',
28:('A family raises glasses together for a toast.',22.5,25,'family','raises','glasses'),
29:'Typography/product slideshow; no unique physical event defensible from observed images.',
31:('A large dragon opens its glowing mouth toward a seated office worker.',21,23.5,'dragon','opens','glowing mouth'),
32:('A hand dips a tortilla chip into a topped tostada.',15,17.5,'hand','dips','tortilla chip'),
33:'Portrait and UI graphic montage; broad campaign/identity descriptions would not define a unique event.',
34:'Static child portraits and text; no defensible bounded action.',
35:('A woman lifts a large pink toy camper toward a man carrying gifts.',23,27,'woman','lifts','pink toy camper'),
36:('A bear-shaped piece of ice melts among larger ice cubes.',29.5,35,'bear-shaped ice','melts','ice cubes'),
37:('Metal pieces assemble into the shape of a butterfly.',15,25,'metal pieces','assemble','butterfly'),
38:('A bright camera flash washes over a family posing for a photograph.',37,39.5,'camera flash','washes over','family'),
39:'Repeated burger turntable shots; no unique supported action beyond generic product rotation.',
40:('A bearded man in a white coat lifts food to his mouth with a fork.',55.5,57.5,'man in white coat','lifts','fork and food'),
42:('A skater wearing a red helmet exits a tunnel into daylight.',25,27.5,'skater','exits','tunnel'),
45:('A pit crew member works on a racing-car wheel with a tool.',13,15.5,'pit crew member','works on','racing-car wheel'),
46:('A mechanic closes the hood of a car beside a race track.',21.5,24,'mechanic','closes','car hood'),
48:('A receipt feeds out of a small printer on a table.',129.5,132.5,'receipt','feeds out','printer'),
49:'Repeated seated interview gestures and portraits; no unique event/action pair identified in bounded inspection.',
50:('A person scrolls a phone screen showing the Take It Down website.',21,25.5,'person','scrolls','phone screen'),
52:('A long narrow boat moves across a river at sunset.',5,7.5,'boat','moves across','river'),
54:'Repeated jewelry portraits and necklace-touching poses; no reliable unique action selected.',
55:'Sale labels and animated phone product cards; no defensible unique physical action established.',
57:'Office conversation repeats views of people and monitors; no distinct action with clear onset and offset selected.',
58:('Two knitted characters hold hands and dance under purple lights.',15,17.5,'knitted characters','dance','hands'),
59:('The neck of a bottle dips into red wax and rises out again.',25,27.5,'bottle neck','dips','red wax'),
60:('A woman runs along a riverside path with a large wheel behind her.',44,49.5,'woman','runs','riverside path'),
61:('A woman fastens the strap of a bicycle helmet under her chin.',21,23.5,'woman','fastens','helmet strap')}
def main():
 p=json.loads((O/'manifests/evaluation_protocol.json').read_text());rows=json.loads((O/'audit/recovery_inventory.json').read_text());out=[]
 assert (O/'manifests/experimental_pipeline_freeze.json').exists()
 for r in rows:
  if r['source_id'] not in p['reserved_offline_source_ids']:continue
  n=int(r['recovery_id'].split('_')[1]);v=NOTES[n];events=[]
  if isinstance(v,tuple):
   q,s,e,subject,action,obj=v
   events=[{'query':q,'coarse_start':s,'coarse_end':e,'visible_subject':subject,'visible_action':action,'visible_object':obj,'actual_visual_support':True,'event_quality':dict(support=4,specificity=4,uniqueness=4,boundary_visibility=3,clarity=4),'generator_evidence_path':r['overview_path'],'generator_evidence_sha256':r['overview_sha256'],'human_verified':False}]
  out.append({'source_id':r['source_id'],'recovery_id':r['recovery_id'],'events':events,'abstention_reason':v if isinstance(v,str) else None,'actual_complete_overview_inspected':True,'model_interface':'Codex multimodal image inspection','model_revision':'NOT_EXPOSED','review_blinding':'Prior task exposure disclosed; frozen experimental code/config before new reserved inspection. AI review remains unblinded.'})
 assert len(out)==41
 (O/'recovery/reserved_events.json').write_text(json.dumps(out,indent=2)+'\n')
 print('SOURCES',len(out),'EVENTS',sum(len(r['events']) for r in out))
if __name__=='__main__':main()
