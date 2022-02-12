// Based on the Vulkano triangle example.

// Triangle example Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::image::ImageCreateFlags;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::{
    view::ImageView, ImageAccess, ImageDimensions, ImageUsage, SwapchainImage, StorageImage
};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint, Pipeline};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// We'll make our Color type an RGBA8888 pixel.
type Color = (u8,u8,u8,u8);

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

#[derive(Copy, Clone)]
struct Position {
    x: usize,
    y: usize
}

impl Position {
    fn incx(&mut self, scalar: usize) {
        self.x += scalar;
    }

    fn decx(&mut self, scalar: usize) {
        self.x -= scalar;
    }
}

// Here's what clear looks like, though we won't use it
#[allow(dead_code)]
fn clear(fb:&mut [Color], c:Color) {
    fb.fill(c);
}

#[allow(dead_code)]
fn hline(fb: &mut [Color], x0: usize, x1: usize, y: usize, col: Color) {
    fb[y*WIDTH + x0..y*WIDTH + x1].fill(col);
}

#[allow(dead_code)]
fn vline(fb: &mut [Color], x: usize, y0: usize, y1: usize, col: Color) {
    for y in y0..y1 {
        fb[y * WIDTH + x..y * WIDTH + x + 1].fill(col) // QUESTION: how to apply fill to single cell
    }
}

#[allow(dead_code)]
fn line(fb: &mut [Color], p0: Position, p1: Position, col: Color) {
    let mut x = p0.x as i64;
    let mut y = p0.y as i64;
    let x0 = p0.x as i64;
    let y0 = p0.y as i64;
    let x1 = p1.x as i64;
    let y1 = p1.y as i64;
    let dx = (x1 - x0).abs();
    let sx: i64 = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy: i64 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    while x != x1 || y != y1 {
        fb[(y as usize * WIDTH + x as usize)
           ..(y as usize * WIDTH + (x as usize + 1))]
            .fill(col);
        let e2 = 2 * err;
        if dy <= e2 {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

#[allow(dead_code)]
fn draw_filled_rect(fb: &mut [Color], p0: Position, p1: Position, col: Color) {
    for y in p0.y..p1.y {
        fb[(y*WIDTH + p0.x)..(y*WIDTH + p1.x)].fill(col);
    }
}

#[allow(dead_code)]
fn draw_outlined_rect(fb: &mut [Color], p0: Position, p1: Position, col: Color) {
    hline(fb, p0.x, p1.x, p0.y, col);
    hline(fb, p0.x, p1.x, p1.y, col);
    vline(fb, p0.x, p0.y, p1.y, col);
    vline(fb, p1.x, p0.y, p1.y, col);
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| {
            p.supported_extensions().is_superset_of(&device_extensions)
        })
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| {
                    q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
                })
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            }
        })
        .unwrap();
    let (device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
        .unwrap();
    let queue = queues.next().unwrap();
    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    // We now create a buffer that will store the shape of our triangle.
    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
        uv: [f32;2]
    }
    vulkano::impl_vertex!(Vertex, position, uv);

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-1.0, -1.0],
                uv: [0.0, 0.0]
            },
            Vertex {
                position: [3.0, -1.0],
                uv: [2.0, 0.0]
            },
            Vertex {
                position: [-1.0, 3.0],
                uv: [0.0, 2.0]
            },
        ]
            .iter()
            .cloned(),
    )
        .unwrap();
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

                layout(location = 0) in vec2 position;
                layout(location = 1) in vec2 uv;
                layout(location = 0) out vec2 out_uv;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    out_uv = uv;
                }
            "
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450

                layout(set = 0, binding = 0) uniform sampler2D tex;
                layout(location = 0) in vec2 uv;
                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = texture(tex, uv);
                }
            "
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();


    // Here's our (2D drawing) framebuffer.
    let mut fb2d = [(128,64,64,255); WIDTH*HEIGHT];
    // We'll work on it locally, and copy it to a GPU buffer every frame.
    // Then on the GPU, we'll copy it into an Image.
    let fb2d_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_source(),
        false,
        (0..WIDTH*HEIGHT).map(|_| (255_u8,0_u8,0_u8,0_u8))
    ).unwrap();
    // Let's set up the Image we'll copy into:
    let dimensions = ImageDimensions::Dim2d {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        array_layers: 1,
    };
    let fb2d_image = StorageImage::with_usage(
        device.clone(),
        dimensions,
        Format::R8G8B8A8_UNORM,
        ImageUsage {
            // This part is key!
            transfer_destination: true,
            sampled: true,
            storage: true,
            transfer_source: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        },
        ImageCreateFlags::default(),
        std::iter::once(queue_family)
    ).unwrap();
    // Get a view on it to use as a texture:
    let fb2d_texture = ImageView::new(fb2d_image.clone()).unwrap();

    let fb2d_sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
        .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                // Pro move: We're going to cover the screen completely. Trust us!
                load: DontCare,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
        .unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();


    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .add_sampled_image(fb2d_texture, fb2d_sampler)
        .unwrap();

    let set = set_builder.build().unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut now_keys = [false; 255];
    let mut prev_keys = now_keys.clone();

    let mut now_lmouse = false;
    let mut prev_lmouse = false;
    
    let colors = [(255,0,0,255), (0,255,0,255), (0,0,255,255)];
    let mut color = 0;

    let x_unit = WIDTH/3;
    let y_unit = HEIGHT/7;

    let mut vel = 1_usize;

    // vertices of outlined rect
    let mut p1 = Position { x: x_unit, y: y_unit };
    let mut p2 = Position { x: 2*x_unit, y: 3*y_unit };
    let mut p3 = Position { x: x_unit, y: 3*y_unit };
    let mut p4 = Position { x: 2*x_unit, y: y_unit };

    // tl and br vertices of filled rect
    let mut p5 = Position { x: x_unit, y: 4*y_unit };
    let mut p6 = Position { x: 2*x_unit, y: 6*y_unit };
    let mut p7 = Position { x: x_unit, y: 6*y_unit };
    let mut p8 = Position { x: 2*x_unit, y: 4*y_unit };

    // SCREENSAVER
    // let mut pos: (i32, i32) = (WIDTH as i32 / 2, HEIGHT as i32 / 2);
    // let mut vel: (i32, i32) = (1, 1);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            },
            Event::NewEvents(_) => {
                // Leave now_keys alone, but copy over all changed keys
                prev_keys.copy_from_slice(&now_keys);
                prev_lmouse = now_lmouse;
            },
            Event::WindowEvent {
                // Note this deeply nested pattern match
                event: WindowEvent::KeyboardInput {
                    input:winit::event::KeyboardInput {
                        // Which serves to filter out only events we actually want
                        virtual_keycode:Some(keycode),
                        state,
                        ..
                    },
                    ..
                },
                ..
            } => {
                // It also binds these handy variable names!
                match state {
                    winit::event::ElementState::Pressed => {
                        // VirtualKeycode is an enum with a defined representation
                        now_keys[keycode as usize] = true;
                    },
                    winit::event::ElementState::Released => {
                        now_keys[keycode as usize] = false;
                    }
                }
            },
            Event::WindowEvent {
                event: WindowEvent::MouseInput {
                    button:btn,
                    state,
                    ..
                },
                ..
            } => {
                match btn {
                    winit::event::MouseButton::Left => {
                        match state {
                            winit::event::ElementState::Pressed => {
                                // VirtualKeycode is an enum with a defined representation
                                now_lmouse = true;
                            },
                            winit::event::ElementState::Released => {
                                now_lmouse = false;
                            }
                        }
                    },
                    _ => {}
                }
            },
            Event::MainEventsCleared => {
                {
                    // We need to synchronize here to send new data to the GPU.
                    // We can't send the new framebuffer until the previous frame is done being drawn.
                    // Dropping the future will block until it's done.
                    if let Some(mut fut) = previous_frame_end.take() {
                        fut.cleanup_finished();
                    }
                }

                // We can actually handle events now that we know what they all are.
                // Mouse Events
                if now_lmouse && !prev_lmouse {
                    std::mem::swap(&mut p1, &mut p5);
                    std::mem::swap(&mut p2, &mut p6);
                    std::mem::swap(&mut p3, &mut p7);
                    std::mem::swap(&mut p4, &mut p8);
                }

                // Keyboard Events
                if now_keys[VirtualKeyCode::Escape as usize] {
                    *control_flow = ControlFlow::Exit;
                }
                if now_keys[VirtualKeyCode::LShift as usize] || now_keys[VirtualKeyCode::RShift as usize] {
                    vel = 2; // Why is there a warning here?
                } else {
                    vel = 1;
                }
                if now_keys[VirtualKeyCode::Up as usize] {
                    color = (color + 1) % colors.len();
                }
                if now_keys[VirtualKeyCode::Down as usize] {
                    // What is this if doing?
                    color = if color == 0 { colors.len() - 1 } else { color - 1 };
                }
                
                if now_keys[VirtualKeyCode::Left as usize] && p2.x - p1.x - vel > 0 && p6.x - p5.x - vel > 0 {
                    
                    p1.incx(vel);
                    p2.decx(vel);
                    p3.incx(vel);
                    p4.decx(vel);
                    p5.incx(vel);
                    p6.decx(vel);
                }
                if now_keys[VirtualKeyCode::Right as usize] && p2.x - p1.x + 2*vel < WIDTH && p6.x - p5.x + 2*vel < WIDTH {
                    p1.decx(vel);
                    p2.incx(vel);
                    p3.decx(vel);
                    p4.incx(vel);
                    p5.decx(vel);
                    p6.incx(vel);
                }
                
                clear(&mut fb2d, (255,255,255,255));

                draw_outlined_rect(&mut fb2d, p1, p2, colors[color]);
                line(&mut fb2d, p1, p2, colors[color]);
                line(&mut fb2d, p3, p4, colors[color]);

                draw_filled_rect(&mut fb2d, p5, p6, colors[color]);

                // SCREENSAVER
                // clear(&mut fb2d, (0,0,0,255));
                // let rect_width: usize = 50;
                // let rect_height: usize = 50;
                // draw_filled_rect(&mut fb2d, 
                //     (pos.0 as usize, pos.1 as usize), 
                //     (rect_width, rect_height), 
                //     (255,255,255,255));
                // if pos.0 as usize + rect_width == WIDTH || pos.0 == 0 {
                //     vel = (-1 * vel.0, vel.1);
                // }
                // if pos.1 as usize + rect_height == HEIGHT || pos.1 == 0 {
                //     vel = (vel.0, -1 * vel.1);
                // }
                // pos = (pos.0 + vel.0, pos.1 + vel.1);

                // Now we can copy into our buffer.
                {
                    let writable_fb = &mut *fb2d_buffer.write().unwrap();
                    writable_fb.copy_from_slice(&fb2d);
                }

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                    .unwrap();

                builder
                // Now copy that framebuffer buffer into the framebuffer image
                    .copy_buffer_to_image(fb2d_buffer.clone(), fb2d_image.clone())
                    .unwrap()
                // And resume our regularly scheduled programming
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        std::iter::once(vulkano::format::ClearValue::None)
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = acquire_future
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}