"""Resource forecasts are estimates, separate from annotation acceptance gates."""
from math import sqrt


def assess_capacity(*, attempted, accepted_p2, accepted_total, pending, unused_candidates,
                    bytes_received, started_epoch, now_epoch, max_runtime_seconds,
                    max_download_bytes, completed_waves, target=500):
    if not 0<=accepted_p2<=attempted or pending<0 or accepted_p2+pending>attempted:
        raise ValueError('Inconsistent observed wave counts')
    elapsed=max(0,now_epoch-started_epoch)
    available_seconds=max(0,max_runtime_seconds-elapsed)
    available_bytes=max(0,max_download_bytes-bytes_received)
    remaining=max(0,target-accepted_total)
    # Pending cases are optimistically treated as future successes ONLY in this
    # forecast; they never enter accepted_total or the actual target calculation.
    effective_remaining=max(0,remaining-pending)
    upper=None;projected_seconds=None;projected_bytes=None;projected_supply=None
    if attempted:
        z=1.96;p=(accepted_p2+pending)/attempted
        upper=min(1,(p+z*z/(2*attempted)+z*sqrt(p*(1-p)/attempted+z*z/(4*attempted**2)))/(1+z*z/attempted))
        projected_seconds=effective_remaining*(elapsed/attempted)/upper
        projected_bytes=effective_remaining*(bytes_received/attempted)/upper
        projected_supply=unused_candidates*upper
    reasons=[]
    if remaining and completed_waves>=2 and attempted>=50:
        if projected_seconds>available_seconds:reasons.append('PROJECTED_ANNOTATION_THROUGHPUT_EXCEEDS_REMAINING_RUNTIME')
        if projected_bytes>available_bytes:reasons.append('PROJECTED_DOWNLOAD_BYTES_EXCEED_REMAINING_BUDGET')
        if projected_supply<effective_remaining:reasons.append('PROJECTED_ACCEPTED_YIELD_IN_UNUSED_POOL_INSUFFICIENT')
    return {'schema':'LUMAE_P2_CAPACITY_FORECAST_V1','classification':'INFERRED_FROM_TWO_BOUNDED_WAVES_NOT_A_PROOF_OF_SOURCE_EXHAUSTION',
        'accepted_unique_new_pairs':accepted_total,'remaining_new_pairs':remaining,'accepted_p2':accepted_p2,'attempted':attempted,
        'pending_not_counted_as_accepted':pending,'completed_waves':completed_waves,'elapsed_seconds':elapsed,
        'remaining_runtime_seconds':available_seconds,'remaining_download_bytes':available_bytes,'unused_candidates':unused_candidates,
        'optimistic_yield_wilson_95_upper_including_pending':upper,'optimistic_projected_seconds_to_target':projected_seconds,
        'optimistic_projected_bytes_to_target':projected_bytes,'optimistic_projected_additional_pairs_from_unused_pool':projected_supply,
        'stop':bool(reasons),'stop_reasons':reasons,'budget_actually_exhausted':available_seconds==0 or available_bytes==0,
        'permission_or_integrity_failure':False,'numeric_pilot20_gate_used':False,
        'assumption':'Seeded candidate order remains representative; same measured process and resource limits. Pending outcomes all assumed positive for optimistic forecast. A changed validated process requires a new versioned capacity assessment.'}
