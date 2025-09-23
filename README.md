# cranexpr

cranexpr is like [`std.Expr`](https://www.vapoursynth.com/doc/functions/video/expr.html)
but built on top of [Cranelift](https://cranelift.dev/). It's a [VapourSynth](https://www.vapoursynth.com/)
plugin that allows one to evaluate an expression per pixel.

## Features

- Arithmetic: `+`, `-`, `*`, `/`, `%`, `pow`, `exp`, `log`, `sqrt`.
- Trigonometry: `sin`, `cos`, `tan`.
- Comparison: `>`, `<`, `max`, `min`.
- Rounding: `floor`, `round`.
- Ternary (if/else): `?`.
- `sgn`: Returns the sign of a value (-1 if negative, 1 if positive, 0 if zero).
- Stack manipulation:
  - `dropN`, `drop`: drops the top N values from the stack. `drop` is equivalent
    to `drop1`.
  - `dupN`, `dup`: allows a value N steps up in the stack to be duplicated. The
    top value of the stack has index 0 meaning that `dup` is equivalent to
    `dup0`.
  - `swapN`, `swap`: allows a value N steps up in the stack to be swapped. The
    top value of the stack has index 0 meaning that `swap` is equivalent to
    `swap1`. This is because `swapN` always swaps with the topmost value at
    index 0.
- Supports any number of input clips. `srcN` may be used to access the `N`-th
  input clip. Shorthand aliases `x`, `y`, `z`, `a`, `b`, `c`, etc. map to
  `src0`, `src1`, `src2`, `src3`, `src4`, `src5`, etc., up to `w` being `src25`.
  Beyond that, use `srcN`.

## API

```python
cranexpr.Expr(
  clips: Sequence[vs.VideoNode],
  expr: SequenceNotStr[str],
  format: int | None = None,
) -> vs.VideoNode
```

- `clips` — Input video nodes.
- `expr` — Reverse Polish Notation (RPN) expression(s) for each plane. The
  expression given for the previous plane is used if the list contains fewer
  expressions than the input clip has planes. This means that a single
  expression will be applied to all planes by default.
- `format` — By default the output format is the same as the first input clip's
  format. This can be overridden by setting this parameter.
