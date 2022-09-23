use std::{
    borrow::Cow,
    ffi::{c_char, CStr},
    ops::Deref,
    sync::{Arc, Mutex},
};

use ash::{extensions::ext::DebugUtils, extensions::khr, vk};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};

pub struct Device {
    device: ash::Device,
    instance: ash::Instance,
    entry: ash::Entry,
    debug_callback: vk::DebugUtilsMessengerEXT,
    pdevice: vk::PhysicalDevice,
    pub allocator: Option<Arc<Mutex<Allocator>>>,

    debug_utils_loader: DebugUtils,

    queue_family_index: u32,
}

impl Device {
    pub fn create() -> Self {
        unsafe {
            let entry = ash::Entry::linked();

            let appname = CStr::from_bytes_with_nul_unchecked(b"VkJit\0");

            let layer_names = [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )];

            let layer_names_raw = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect::<Vec<*const c_char>>();

            let mut extension_names = vec![
                DebugUtils::name().as_ptr(),
                CStr::from_bytes_with_nul_unchecked(b"VK_KHR_get_physical_device_properties2\0")
                    .as_ptr(),
                vk::KhrDeviceGroupCreationFn::name().as_ptr(),
            ];

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(appname)
                .application_version(0)
                .engine_name(appname)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 0, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(&extension_names);

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Could not create instance");

            let fp_v1_1 = instance.fp_v1_1();
            let get_physical_device_features2 = fp_v1_1.get_physical_device_features2;
            let get_physical_device_properties2 = fp_v1_1.get_physical_device_properties2;

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);

            let debug_callback = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");

            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_compute = info
                                .queue_flags
                                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER);
                            if supports_compute {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Could not find suitable device!");

            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw =
                [CStr::from_bytes_with_nul_unchecked(b"VK_KHR_device_group\0").as_ptr()];

            let mut buffer_physical_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true);

            let mut descriptor_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeatures::builder();

            // Features2
            let mut features2 = vk::PhysicalDeviceFeatures2::builder()
                .push_next(&mut buffer_physical_device_address_features)
                .push_next(&mut descriptor_indexing_features);

            let mut features2 = features2.build();

            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut features2);

            let device = instance
                .create_device(pdevice, &device_create_info, None)
                .expect("Could not create device!");

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                debug_settings: AllocatorDebugSettings {
                    log_leaks_on_shutdown: true,
                    log_memory_information: true,
                    log_allocations: true,
                    ..Default::default()
                },
                buffer_device_address: true,
            })
            .expect("Could not create allocator");
            Self {
                device,
                instance,
                entry,
                debug_callback,
                pdevice,
                allocator: Some(Arc::new(Mutex::new(allocator))),

                debug_utils_loader,

                queue_family_index,
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.allocator.take();
            self.device.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}
