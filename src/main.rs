mod klir;

fn main() {
    let mut ir = klir::Ir::default();
    let x = ir.one();
    let y = ir.one();
    let z = ir.add(x, y);
    println!("{:?}", ir)
}
