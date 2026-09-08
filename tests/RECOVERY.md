# Controlled recovery qualification

These tests interrupt service or change host state. Run only in an explicitly approved
maintenance/test window with clients paused. Record commands, timestamps, release IDs,
observations and sanitized evidence in `reports/`. Do not claim a pass from a checklist.

1. **Offline artifact restart:** block model-host access for the test container using
   an operator-controlled network rule, recreate the service, and verify model lookup,
   generation and tools. Startup sets offline library flags and never pulls an image;
   independently verify that model-host unavailability does not prevent startup.
2. **Warm recreation:** retain cache directory metadata before/after recreation and
   inspect engine compilation-cache messages. Weight reuse is distinct from compilation
   reuse and from prefix-cache acceleration.
3. **Process exit:** while testing the Docker `unless-stopped` policy, terminate only
   this project's inference process. Verify a restart-count increment, health and a
   generation. Restore candidate policy `no` after the isolated policy test if it has
   not yet been accepted. Never target unrelated GPU processes.
4. **Host reboot:** with this service running under the intended restart policy, reboot
   in an approved window, then verify inference. Repeat after an intentional stop and
   confirm it stays stopped. Record both cases.
5. **Failed replacement:** retain a known accepted release, prepare a deliberately
   invalid runtime candidate, and attempt `upgrade --release <id>`. Verify nonzero exit,
   preserved failure diagnostics and successful inference from the accepted release.
   Separately verify invalid configuration fails before stopping the active service.
6. **Interrupted preparation:** interrupt a candidate download; active configuration
   and accepted artifacts must remain unchanged. Rerun preparation and verify integrity.
7. **Unhealthy process:** induce health failure only for the test service and inspect
   status. Confirm unhealthy is reported without claiming automatic health repair.
8. **Mutation race:** hold `state/mutation.lock` with `flock`, invoke a mutation and
   confirm immediate rejection. During candidate replacement, a second mutation must
   not create another engine or change release state.
9. **Memory / soak:** inspect each long-context report and the eight-hour soak report.
   Require >=16 GiB minimum MemAvailable, no host OOM or sustained swapping, at least
   100 short requests, tool rounds and three long requests, and no unexplained restart.

Collect these results in a reviewed evidence JSON with the exact active release ID.
The acceptance command checks the explicit operator attestations plus automated API,
context, benchmark and soak reports. It cannot independently prove a reboot or external
network-isolation test from an operator assertion.
