#pragma once

#include <string>

namespace runtime{

    enum class StatusCode{
        Ok,
        InvalidArgument,
        InvalidShape,
        InvalidDataType,
        InvalidDevice,
        OutOfMemory,
        NotImplemented,
        InternalError,
        AlreadyExist,
        NotFound
    };

    class Status{

        public:
            Status() = delete;
            Status(StatusCode code);
            Status(StatusCode code, std::string message);

            bool ok() const noexcept;
            StatusCode code() const noexcept;
            const std::string& message() const noexcept;

            static Status OK();
            static Status Error(StatusCode code, std::string message);

        private:
            StatusCode code_ = StatusCode::Ok;
            std::string message_;

    };

}