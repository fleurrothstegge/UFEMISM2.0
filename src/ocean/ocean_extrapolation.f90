module ocean_extrapolation

  use precisions, only: dp
  use control_resources_and_error_messaging, only: init_routine, finalise_routine
  use model_configuration, only: C
  use mesh_types, only: type_mesh
  use ice_model_types, only: type_ice_model
  use mesh_utilities, only: extrapolate_Gaussian

  implicit none

  private

  public :: extrapolate_ocean_forcing

contains

  subroutine extrapolate_ocean_forcing( mesh, ice, d_partial)
    ! Extrapolate offshore ocean properties into full domain

    ! In/output variables
    type(type_mesh),                                   intent(in)  :: mesh
    type(type_ice_model),                              intent(in)  :: ice
    real(dp), dimension(mesh%vi1:mesh%vi2,C%nz_ocean), intent(out) :: d_partial

    ! Local variables
    character(len=1024), parameter        :: routine_name = 'extrapolate_ocean_forcing'
    integer                               :: vi, k
    integer, dimension(mesh%vi1:mesh%vi2) :: mask_fill
    real(dp), parameter                   :: sigma = 4e4

    ! Add routine to path
    call init_routine( routine_name)

    ! == Step 1: extrapolate horizontally into cavity ==

    do k = 1, C%nz_ocean
      ! Initialise assuming there's valid data everywhere
      mask_fill = 2
      ! Check for NaNs in cavity
      do vi = mesh%vi1, mesh%vi2
        ! Check for NaNs
        if (d_partial( vi, k) /= d_partial( vi, k)) then
          ! Check whether in cavity
          if ((C%z_ocean( k) > -ice%Hib( vi)) .and. (C%z_ocean( k) < -ice%Hb( vi))) then
            ! In cavity, so extrapolate here
            mask_fill( vi) = 1
          else
            ! Not in cavity, don't extrapolate here
            mask_fill( vi) = 0
          end if
        end if
      end do
      ! Fill NaN vertices within this layer
      call extrapolate_Gaussian( mesh, mask_fill, d_partial(:,k), sigma)
    end do

    ! == Step 2: extrapolate vertically into ice shelf and bedrock ==

    ! == Step 3: extrapolate horizontally everywhere ==

    ! Extrapolate into NaN areas independently for each layer
   ! do k = 1, C%nz_ocean
      ! Initialise assuming there's valid data everywhere
   !   mask_fill = 2
      ! Check this mesh layer for NaNs
   !   do vi = mesh%vi1, mesh%vi2
   !     if (d_partial( vi,k) /= d_partial( vi,k)) then
          ! if NaN, allow extrapolation here
   !       mask_fill( vi) = 1
   !     end if
   !   end do
      ! Fill NaN vertices within this layer
   !   call extrapolate_Gaussian( mesh, mask_fill, d_partial(:,k), sigma)
   ! end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine extrapolate_ocean_forcing

end module ocean_extrapolation
