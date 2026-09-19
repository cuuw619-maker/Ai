#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    pub new_id: u32,
}
