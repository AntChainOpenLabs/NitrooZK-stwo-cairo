
use itertools::Itertools;
use stwo_cairo_adapter::memory::Memory;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::bindings_airs;

pub type InputType = M31;
pub type PackedInputType = PackedM31;
pub type CudaPackedInputType = [BaseFieldVec ;1];
/// A struct that represents a mapping from Address to ID. Zero address is not allowed.
#[derive(Debug)]
pub struct CudaAddressToId {
    /// Since zero address is reserved, the vector holding the data is offset by 1, i.e. the ID of
    /// address 1 is stored at index 0, and so on.
    data: Uint32Vec,
}
impl CudaAddressToId {
    pub fn new(data: Uint32Vec) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}



/// A struct to generate the memory address to ID trace.
pub struct CudaClaimGenerator {
    pub address_to_raw_id: Uint32Vec,
    pub multiplicities: Uint32Vec,
}
impl CudaClaimGenerator {
    pub fn new(memory: &Memory) -> Self {
        // Note that while `memory.address_to_id` starts from address 0, the memory component can
        // only yield addresses starting from 1.
        let mem_vec = (1..memory.address_to_id.len())
                .map(|addr| memory.get_raw_id(addr as u32))
                .collect_vec();
        let address_to_raw_id = Uint32Vec::from_vec(mem_vec);
        let multiplicities = Uint32Vec::new_uninitialized(address_to_raw_id.size);
        // println!("address_to_raw_id size: {}", address_to_raw_id.size);
        // println!("cuda address_to_raw_id : {:?}", address_to_raw_id.to_vec());

        Self {
            address_to_raw_id,
            multiplicities,
        }
    }

    pub fn add_cuda_input(&mut self, addr: &BaseField) {
        self.multiplicities.increase_at(addr.0 - 1);
    }

    pub fn add_cuda_inputs(&mut self, cuda_inputs: &[CudaPackedInputType]) {
        let inputs_vec = cuda_inputs.iter()
            .flat_map(|row| row.iter().map(|x| x.device_ptr))
            .collect_vec();
        // println!("memory_address_to_id_add_inputs col len: {:?}, row len:{:?}", cuda_inputs.len(), cuda_inputs[0][0].size);

        unsafe {
            bindings_airs::memory_address_to_id_add_inputs(
                inputs_vec.as_ptr(),
                cuda_inputs.len() as u32,
                cuda_inputs[0][0].size as u32,
                self.multiplicities.device_ptr,
                self.multiplicities.size.ilog2(),
            );
        }
    }


    pub fn get_id(&self, input: BaseField) -> M31 {
        let id = self.address_to_raw_id.get_data(input.0 as usize);
        M31 (id)
    }

    pub fn get_id_vec(&self, input_vec: Vec<BaseField>) -> BaseFieldVec {
        let id_vec = input_vec
            .iter()
            .map(|x| M31 (self.address_to_raw_id.get_data(x.0 as usize)))
            .collect::<Vec<_>>();
        BaseFieldVec::from_vec(id_vec)
    }

}


