module component_tests

  ! All the component tests.

  use mpi
  use precisions                                             , only: dp
  use mpi_basic                                              , only: par, cerr, ierr, recv_status, sync
  use control_resources_and_error_messaging                  , only: warning, crash, happy, init_routine, finalise_routine, colour_string
  use model_configuration                                    , only: C
  use component_tests_create_test_meshes                     , only: create_all_test_meshes

  implicit none

  private

  public :: run_all_component_tests

contains

  subroutine run_all_component_tests

    ! Local variables:
    character(len=256), parameter :: routine_name = 'run_all_component_tests'

    ! Add routine to path
    call init_routine( routine_name)

    if (par%master) write(0,'(a)') ''
    if (par%master) write(0,'(a)') ' Running UFEMISM component tests...'

    ! Create an output folder and output file
    call create_component_tests_output_folder

    ! Create the suite of test meshes for the component tests.
    call create_all_test_meshes

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine run_all_component_tests

  !> Create the component test output folder
  subroutine create_component_tests_output_folder

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'create_component_tests_output_folder'
    logical                        :: ex
    character(len=1024)            :: cwd

    ! Add routine to path
    call init_routine( routine_name)

    C%output_dir = 'results_component_tests'

    ! Create the directory
    if (par%master) then

      ! Remove existing folder if necessary
      inquire( file = trim( C%output_dir) // '/.', exist = ex)
      if (ex) then
        call system('rm -rf ' // trim( C%output_dir))
      end if

      ! Create output directory
      CALL system('mkdir ' // trim( C%output_dir))

      ! Tell the user where it is
      call getcwd( cwd)
      write(0,'(A)') ''
      write(0,'(A)') ' Output directory: ' // colour_string( trim(cwd)//'/'//trim( C%output_dir), 'light blue')
      write(0,'(A)') ''

    end if
    call sync

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine create_component_tests_output_folder

end module component_tests
