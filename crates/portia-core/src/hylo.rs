//! Stack-safe `MappableFrame`, `Collapsible`, and `Expandable` recursion schemes powered by
//! the [`recursion`](https://crates.io/crates/recursion) crate.
//!
//! Re-exports the canonical traits from [`recursion`] (`PartiallyApplied`, `MappableFrame`,
//! `Collapsible`, `CollapsibleExt`, `Expandable`, `ExpandableExt`) and provides
//! [`expand_and_collapse`], [`try_expand_and_collapse`], and [`RoseFrame`] / [`RoseTree`]
//! delegated directly to [`recursion::CollapsibleExt`] and [`recursion::ExpandableExt`].

use std::cell::RefCell;
use std::marker::PhantomData;

pub use recursion::{
    Collapsible, CollapsibleExt, Expandable, ExpandableExt, MappableFrame, PartiallyApplied,
};

/// Coalgebraic seed adapter `(Seed, FnMut(Seed) -> F::Frame<Seed>)` implementing
/// [`recursion::Collapsible`] so [`expand_and_collapse`] executes directly via
/// [`recursion::CollapsibleExt::collapse_frames`].
struct CoalgebraSeed<'a, F: MappableFrame, Seed, ExpandFn> {
    seed: Seed,
    expand: &'a RefCell<ExpandFn>,
    _frame: PhantomData<F>,
}

impl<'a, F, Seed, ExpandFn> Collapsible for CoalgebraSeed<'a, F, Seed, ExpandFn>
where
    F: MappableFrame,
    ExpandFn: FnMut(Seed) -> F::Frame<Seed>,
{
    type FrameToken = F;

    fn into_frame(self) -> <Self::FrameToken as MappableFrame>::Frame<Self> {
        let raw_frame = (self.expand.borrow_mut())(self.seed);
        F::map_frame(raw_frame, |child_seed| CoalgebraSeed {
            seed: child_seed,
            expand: self.expand,
            _frame: PhantomData,
        })
    }
}

/// Functorial coproduct frame `Result<F::Frame<X>, E>` lifting any [`recursion::MappableFrame`] `F`
/// so fallible coalgebras (`Seed -> Result<F::Frame<Seed>, E>`) execute directly on
/// [`recursion::CollapsibleExt::try_collapse_frames`].
pub struct FallibleFrame<F: MappableFrame, E, X> {
    pub inner: Result<F::Frame<X>, E>,
}

impl<F: MappableFrame, E> MappableFrame for FallibleFrame<F, E, PartiallyApplied> {
    type Frame<X> = FallibleFrame<F, E, X>;

    fn map_frame<A, B>(input: Self::Frame<A>, f: impl FnMut(A) -> B) -> Self::Frame<B> {
        FallibleFrame {
            inner: input.inner.map(|frame| F::map_frame(frame, f)),
        }
    }
}

struct FallibleCoalgebraSeed<'a, F: MappableFrame, Seed, E, ExpandFn> {
    seed: Seed,
    expand: &'a RefCell<ExpandFn>,
    _frame: PhantomData<(F, E)>,
}

impl<'a, F, Seed, E, ExpandFn> Collapsible for FallibleCoalgebraSeed<'a, F, Seed, E, ExpandFn>
where
    F: MappableFrame,
    ExpandFn: FnMut(Seed) -> Result<F::Frame<Seed>, E>,
{
    type FrameToken = FallibleFrame<F, E, PartiallyApplied>;

    fn into_frame(self) -> FallibleFrame<F, E, Self> {
        let res = (self.expand.borrow_mut())(self.seed);
        FallibleFrame {
            inner: res.map(|frame| {
                F::map_frame(frame, |child_seed| FallibleCoalgebraSeed {
                    seed: child_seed,
                    expand: self.expand,
                    _frame: PhantomData,
                })
            }),
        }
    }
}

/// Stack-safe depth-first hylomorphism (`Seed -> F::Frame<Seed>` coalgebra followed by
/// `F::Frame<Out> -> Out` algebra) executed via the [`recursion::CollapsibleExt::collapse_frames`].
pub fn expand_and_collapse<F: MappableFrame, Seed, Out>(
    seed: Seed,
    expand_frame: impl FnMut(Seed) -> F::Frame<Seed>,
    collapse_frame: impl FnMut(F::Frame<Out>) -> Out,
) -> Out {
    let expand_cell = RefCell::new(expand_frame);
    CoalgebraSeed::<F, Seed, _> {
        seed,
        expand: &expand_cell,
        _frame: PhantomData,
    }
    .collapse_frames(collapse_frame)
}

/// Fallible stack-safe depth-first hylomorphism (`Seed -> Result<F::Frame<Seed>, E>` and
/// `F::Frame<Out> -> Result<Out, E>`) executed via the
/// [`recursion::CollapsibleExt::try_collapse_frames`].
pub fn try_expand_and_collapse<F: MappableFrame, Seed, Out, E>(
    seed: Seed,
    expand_frame: impl FnMut(Seed) -> Result<F::Frame<Seed>, E>,
    mut collapse_frame: impl FnMut(F::Frame<Out>) -> Result<Out, E>,
) -> Result<Out, E> {
    let expand_cell = RefCell::new(expand_frame);
    FallibleCoalgebraSeed::<F, Seed, E, _> {
        seed,
        expand: &expand_cell,
        _frame: PhantomData,
    }
    .try_collapse_frames(|fallible_frame| collapse_frame(fallible_frame.inner?))
}

/// Concrete rose tree unfolded via [`recursion::Expandable`] and collapsed via
/// [`recursion::Collapsible`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoseTree<Label> {
    pub label: Label,
    pub children: Vec<RoseTree<Label>>,
}

/// Generic rose-tree frame `RoseFrame<Label, Child>` implementing [`recursion::MappableFrame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoseFrame<Label, Child> {
    pub label: Label,
    pub children: Vec<Child>,
}

impl<Label> MappableFrame for RoseFrame<Label, PartiallyApplied> {
    type Frame<X> = RoseFrame<Label, X>;

    fn map_frame<A, B>(input: Self::Frame<A>, mut f: impl FnMut(A) -> B) -> Self::Frame<B> {
        RoseFrame {
            label: input.label,
            children: input.children.into_iter().map(&mut f).collect(),
        }
    }
}

impl<Label> Expandable for RoseTree<Label> {
    type FrameToken = RoseFrame<Label, PartiallyApplied>;

    fn from_frame(val: RoseFrame<Label, Self>) -> Self {
        RoseTree {
            label: val.label,
            children: val.children,
        }
    }
}

impl<Label> Collapsible for RoseTree<Label> {
    type FrameToken = RoseFrame<Label, PartiallyApplied>;

    fn into_frame(self) -> RoseFrame<Label, Self> {
        RoseFrame {
            label: self.label,
            children: self.children,
        }
    }
}

impl<Label: Clone> Collapsible for &RoseTree<Label> {
    type FrameToken = RoseFrame<Label, PartiallyApplied>;

    fn into_frame(self) -> RoseFrame<Label, Self> {
        RoseFrame {
            label: self.label.clone(),
            children: self.children.iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_safe_hylo_evaluates_deep_chain_without_overflow() {
        // Depth 25,000 linear chain evaluated through the `recursion` crate:
        let depth = 25_000usize;
        let sum = expand_and_collapse::<RoseFrame<usize, PartiallyApplied>, usize, usize>(
            depth,
            |n| {
                if n == 0 {
                    RoseFrame {
                        label: 1,
                        children: vec![],
                    }
                } else {
                    RoseFrame {
                        label: 1,
                        children: vec![n - 1],
                    }
                }
            },
            |frame| frame.label + frame.children.into_iter().sum::<usize>(),
        );
        assert_eq!(sum, depth + 1);

        // Also verify `RoseTree::expand_frames`, `collapse_frames`, and `try_expand_and_collapse`:
        let tree = RoseTree::expand_frames(4usize, |n| RoseFrame {
            label: n,
            children: if n == 0 { vec![] } else { vec![n - 1] },
        });
        let total = (&tree)
            .collapse_frames(|frame| frame.label + frame.children.into_iter().sum::<usize>());
        assert_eq!(total, 10);

        let fallible_ok = try_expand_and_collapse::<
            RoseFrame<usize, PartiallyApplied>,
            usize,
            usize,
            &'static str,
        >(
            4usize,
            |n| {
                Ok(RoseFrame {
                    label: n,
                    children: if n == 0 { vec![] } else { vec![n - 1] },
                })
            },
            |frame| Ok(frame.label + frame.children.into_iter().sum::<usize>()),
        );
        assert_eq!(fallible_ok, Ok(10));

        let fallible_err = try_expand_and_collapse::<
            RoseFrame<usize, PartiallyApplied>,
            usize,
            usize,
            &'static str,
        >(
            4usize,
            |n| {
                if n == 2 {
                    Err("halt at depth 2")
                } else {
                    Ok(RoseFrame {
                        label: n,
                        children: if n == 0 { vec![] } else { vec![n - 1] },
                    })
                }
            },
            |frame| Ok(frame.label + frame.children.into_iter().sum::<usize>()),
        );
        assert_eq!(fallible_err, Err("halt at depth 2"));
    }
}
