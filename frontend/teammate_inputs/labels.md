# Review labels

The CSV currently uses these `manual_verif` values (counts from the demo
file `tuxedni_results_1stAL_round_rev.csv`):

| value        | count |
|--------------|-------|
| `""` (empty) |   728 |
| `Noise`      |   308 |
| `off_effort` |    52 |
| `Beluga`     |     6 |

Note the inconsistent casing in the existing data (`Noise`/`Beluga` are
Title Case but `off_effort` is lower case). Decide on the canonical form
you want going forward — the UI will normalize to whatever you pick here.

## Labels you want available in the review UI

Keep this list short — too many options slow down review. Mark `[x]` for
keep, edit the names to your preferred wording, add any new ones.

- [x] Noise
- [x] Beluga
- [x] off_effort
- [ ] Humpback
- [ ] Orca
- [ ] Unsure
- [ ] (other — fill in)

## Notes (optional)
- Anything to flag about how you use these labels?
