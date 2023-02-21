use std::collections::HashSet;

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
