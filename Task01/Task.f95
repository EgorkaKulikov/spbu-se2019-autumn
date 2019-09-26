program SolverOfSystemOfLinearEquations
  use omp_lib

  integer(2) :: size
  real(16), allocatable, dimension(:,:) :: matrix
  real(16) :: ratio
  
  read*, size
  allocate (matrix(size+1, size))
  
  read(*,*) matrix

  S1: do i=1,size  

    !$omp parallel do private(ratio, j, k)
    S2: do j=1,size

      if (j /= i .and. 0 /= matrix(i,i))
        
        then ratio = matrix(i, j) / matrix(i, i) 
        S3: do k=i,size+1
          
          matrix(k, j) = matrix(k, j) - matrix(k, i) * ratio
        end do S3
      end if
    end do S2
    !$omp end parallel do
  end do S1
  
  P: do i=1,size

    print*, nint(real(matrix(size+1, i)/matrix(i,i)))
  end do P

  deallocate (matrix)
end
