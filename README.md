# cranexpr

cranexpr is like [`std.Expr`](https://www.vapoursynth.com/doc/functions/video/expr.html)
but built on top of [Cranelift](https://cranelift.dev/). It's a [VapourSynth](https://www.vapoursynth.com/)
plugin that allows one to evaluate an expression per pixel.

## Features

- Arithmetic: `+`, `-`, `*`, `/`, `%`, `pow`, `exp`, `log`, `sqrt`.
- Trigonometry: `sin`, `cos`, `tan`, `atan2`.
- Comparison: `>`, `<`.
- Bitwise: `bitand`, `bitor`, `bitxor`, `bitnot`.
- Clamping: `min`, `max`, `clip` (alias: `clamp`).
- Rounding: `floor`, `round`.
- Ternary (if/else): `?`.
- `sgn`: Returns the sign of a value (-1 if negative, 1 if positive, 0 if zero).
- Constants:
  - `width`: Width of the plane.
  - `height`: Height of the plane.
  - `pi`: π.
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
- Variables:
  - `var!`: Pops the top value from the stack and stores it in a variable named
    `var`.
  - `var@`: Pushes the value of the variable `var` onto the stack.
- Relative pixel access: `clip[relX, relY]:[mode]`.
  - Accesses a pixel relative to the current coordinate (`X`, `Y`). `relX` and
    `relY` must be integer constants.
  - If no suffix is provided, the edge behavior is determined by the filter's
    `boundary` parameter.
    - `:c`: Forces clamped boundary.
    - `:m`: Forces mirrored boundary.
- Absolute pixel access: `absX absY clip[]:[mode]`.
  - Accesses a pixel at an absolute coordinate. It pops `absY` then `absX` from
    the stack. These coordinates can be computed by expressions.
  - If the coordinates are not integers, they will be rounded half to even.
  - **Example:** `X 2 / Y x[]` reads the pixel at half the current X
    coordinate from the first clip, using the default clamp mode.
  - **Boundary Suffixes:**
    - `:c`: Forces clamped boundary.
    - `:m`: Forces mirrored boundary.
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
  boundary: Literal[0, 1] = 0,
) -> vs.VideoNode
```

- `clips` — Input video nodes.
- `expr` — Reverse Polish Notation (RPN) expression(s) for each plane. The
  expression given for the previous plane is used if the list contains fewer
  expressions than the input clip has planes. This means that a single
  expression will be applied to all planes by default.
- `format` — By default the output format is the same as the first input clip's
  format. This can be overridden by setting this parameter.
- `boundary` — Boundary mode. `0` for clamping, `1` for mirroring.
