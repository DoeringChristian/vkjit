use std::collections::HashSet;

use crate::internal::Var;
use crate::{Ir, VarId};

pub struct DepIterator<'a> {
    pub ir: &'a Ir,
    pub stack: Vec<VarId>,
    pub discovered: HashSet<VarId>,
}

impl<'a> Iterator for DepIterator<'a> {
    type Item = VarId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(id) = self.stack.pop() {
            if !self.discovered.contains(&id) {
                let var = self.ir.var(id);
                for id in var.deps.iter().rev() {
                    if !self.discovered.contains(id) {
                        self.stack.push(*id);
                    }
                }
                self.discovered.insert(id);
                return Some(id);
            }
        }
        None
    }
}

pub struct SeIterator<'a> {
    pub ir: &'a Ir,
    pub stack: Vec<VarId>,
    pub discovered: HashSet<VarId>,
}

impl<'a> Iterator for SeIterator<'a> {
    type Item = VarId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(id) = self.stack.pop() {
            if !self.discovered.contains(&id) {
                let var = self.ir.var(id);
                for id in var.side_effects.iter().rev() {
                    if !self.discovered.contains(id) {
                        self.stack.push(*id);
                    }
                }
                for id in var.deps.iter().rev() {
                    if !self.discovered.contains(id) {
                        self.stack.push(*id);
                    }
                }
                self.discovered.insert(id);
                return Some(id);
            }
        }
        None
    }
}

pub struct MutSeVisitor<'a> {
    pub ir: &'a mut Ir,
    pub discovered: HashSet<VarId>,
}

impl<'a> MutSeVisitor<'a> {
    pub fn visit(&mut self, id: VarId, f: &impl Fn(&mut Ir, VarId) -> bool) {
        if !self.discovered.contains(&id) {
            let var = self.ir.var(id);
            let refs = var
                .deps
                .iter()
                .chain(var.side_effects.iter())
                .map(|id| *id)
                .collect::<Vec<_>>();

            if f(self.ir, id) {
                for id in refs {
                    self.visit(id, f);
                }
            }

            self.discovered.insert(id);
        }
    }
}
