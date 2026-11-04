from data_process.lumae_scale.wave_capacity import assess_capacity


def assess(**updates):
    values=dict(attempted=50,accepted_p2=10,accepted_total=17,pending=6,unused_candidates=1037,
        bytes_received=951401435,started_epoch=1000,now_epoch=4000,max_runtime_seconds=28800,
        max_download_bytes=20*2**30,completed_waves=2)
    return assess_capacity(**(values|updates))


def test_forecast_never_counts_pending_or_downloaded_sources_as_accepted():
    r=assess()
    assert r['accepted_unique_new_pairs']==17 and r['remaining_new_pairs']==483
    assert r['pending_not_counted_as_accepted']==6
    assert r['stop'] and not r['budget_actually_exhausted']
    assert 'PROJECTED_ANNOTATION_THROUGHPUT_EXCEEDS_REMAINING_RUNTIME' in r['stop_reasons']
    assert r['numeric_pilot20_gate_used'] is False


def test_low_pilot_yield_does_not_block_first_bounded_wave():
    assert not assess(attempted=25,accepted_p2=0,accepted_total=7,pending=0,completed_waves=1)['stop']


def test_same_budget_clock_survives_resume_and_complete_target_needs_no_more_sources():
    r=assess(now_epoch=11000)
    assert r['elapsed_seconds']==10000 and r['remaining_runtime_seconds']==18800
    assert not assess(accepted_total=500)['stop']


def test_fast_high_yield_process_is_allowed_to_continue():
    r=assess(accepted_p2=45,accepted_total=52,pending=0,now_epoch=1100)
    assert not r['stop']
